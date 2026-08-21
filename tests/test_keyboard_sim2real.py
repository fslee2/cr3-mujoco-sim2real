import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import mujoco
from mujoco.glfw import glfw
import numpy as np

import run_keyboard_sim2real as app
import run_keyboard_sim2real_gui as gui
from cr3_sim2real.hardware import (
    DEFAULT_LIVE_JOINT_SPEED_DEG_S,
    PACKET_MARKER,
    LIVE_SERVO_PERIOD_S,
    LIVE_TARGET_FILTER_FAST_ALPHA,
    LIVE_TARGET_FILTER_SLOW_ALPHA,
    STRICT_20_PERCENT_LIMIT_DEG_S,
    LiveServoHardware,
    FeedbackReceiver,
    MyType,
    RobotFeedback,
    PlaybackHardware,
    clear_robot_error,
    continue_robot_queue,
    get_robot_error_ids,
    joint_target_reached,
    limit_joint_velocity,
    pause_robot_queue,
    disable_robot,
    power_on_robot,
    require_command_success,
    require_motion_ready,
    set_robot_speed_factor,
    start_drag_mode,
    stop_drag_mode,
    stream_live_target_until_reached,
    wait_for_drag_state,
)
from cr3_sim2real.hamer_bridge import (
    HamerArmMapper,
    HamerBridgeTracker,
    hamer_cam_t,
)
from cr3_sim2real.hamer_real import (
    HamerRealCommandPreview,
    HamerRealSafetyConfig,
)
from cr3_sim2real.joint_mapping import real_deg_to_sim_rad, sim_rad_to_real_deg
from cr3_sim2real.quest_hand import (
    QUEST_MOTION_MODE_ORIGINAL,
    QUEST_MOTION_MODE_REVERSED_END,
    QuestHandReceiver,
    QuestWristMapper,
    parse_quest_line,
    quest_quaternion_to_robot_rotation,
)
from cr3_sim2real.trajectory import (
    TrajectoryPoint,
    TrajectoryRecorder,
    build_jointmovj_dry_run,
    build_servoj_dry_run,
    downsample_points,
    resample_joint_trajectory,
    validate_trajectory,
)


class KeyboardSim2RealTests(unittest.TestCase):
    def test_quest_parser_accepts_original_and_debug_headers(self):
        wrist = parse_quest_line(
            "Right wrist:, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.97"
        )
        self.assertIsNotNone(wrist)
        self.assertEqual(wrist.side, "right")
        self.assertEqual(wrist.kind, "wrist")
        np.testing.assert_allclose(wrist.values[:3], [0.1, 0.2, 0.3])

        values = ", ".join(str(value / 100.0) for value in range(63))
        landmarks = parse_quest_line(
            f"Left landmarks | f = 42 | t = 123456789:,{values}"
        )
        self.assertIsNotNone(landmarks)
        self.assertEqual(landmarks.frame_id, 42)
        self.assertEqual(landmarks.device_timestamp_ns, 123456789)
        self.assertEqual(landmarks.values.shape, (21, 3))

    def test_quest_receiver_combines_original_repository_lines(self):
        receiver = QuestHandReceiver()
        receiver._accept_text(
            "Right wrist:, 0.1, 0.2, 0.3, 0, 0, 0, 1",
            "192.0.2.20:9000",
        )
        wrist_only = receiver.get("right")
        receiver._accept_text(
            "Right landmarks:, " + ", ".join(["0"] * 63),
            "192.0.2.20:9000",
        )
        snapshot = receiver.get("right")
        self.assertTrue(snapshot.has_wrist)
        np.testing.assert_allclose(snapshot.wrist_position, [0.1, 0.2, 0.3])
        self.assertEqual(snapshot.landmarks.shape, (21, 3))
        self.assertEqual(snapshot.sender, "192.0.2.20:9000")
        self.assertEqual(wrist_only.wrist_sequence, 1)
        self.assertEqual(snapshot.wrist_sequence, 1)
        self.assertEqual(snapshot.landmarks_sequence, 1)
        self.assertGreater(snapshot.sequence, wrist_only.sequence)

    def test_quest_wrist_mapper_uses_unity_to_robot_axes(self):
        mapper = QuestWristMapper(
            gain=1.0,
            max_delta_m=1.0,
            deadzone_m=0.0,
            ema_alpha=1.0,
        )
        mapper.calibrate([1.0, 2.0, 3.0], [0.4, 0.5, 0.6])
        target = mapper.target_pos([1.1, 2.2, 3.3])
        # Unity (right, up, forward) -> robot (forward, left, up).
        np.testing.assert_allclose(target, [0.7, 0.4, 0.8])

    def test_quest_wrist_mapper_supports_reversed_end_mapping(self):
        mapper = QuestWristMapper(
            gain=1.0,
            max_delta_m=1.0,
            deadzone_m=0.0,
            ema_alpha=1.0,
            motion_mode=QUEST_MOTION_MODE_REVERSED_END,
        )
        mapper.calibrate([1.0, 2.0, 3.0], [0.4, 0.5, 0.6])
        target = mapper.target_pos([1.1, 2.2, 3.3])
        # The flipped tool frame keeps robot X and reverses robot Y/Z.
        np.testing.assert_allclose(target, [0.7, 0.6, 0.4])

    def test_quest_adaptive_filter_reacts_faster_to_fast_motion(self):
        mapper = QuestWristMapper(
            gain=1.0,
            max_delta_m=1.0,
            deadzone_m=0.0,
            ema_alpha=0.20,
            fast_ema_alpha=0.90,
            slow_speed_m_s=0.02,
            fast_speed_m_s=0.20,
            filter_reference_hz=60.0,
        )
        mapper.calibrate([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], timestamp=1.0)
        slow_target = mapper.target_pos(
            [0.0, 0.0, 0.0001], timestamp=1.0 + 1.0 / 60.0
        )
        slow_alpha = mapper.last_alpha

        mapper.calibrate([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], timestamp=2.0)
        fast_target = mapper.target_pos(
            [0.0, 0.0, 0.01], timestamp=2.0 + 1.0 / 60.0
        )
        fast_alpha = mapper.last_alpha

        self.assertLess(slow_alpha, fast_alpha)
        self.assertLess(slow_target[0] / 0.0001, fast_target[0] / 0.01)
        self.assertAlmostEqual(mapper.last_sample_hz, 60.0, places=5)

    def test_quest_quaternion_conversion_preserves_identity_and_axes(self):
        identity = quest_quaternion_to_robot_rotation([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-12)
        # A Unity +90° yaw about its up axis becomes the corresponding
        # conjugated rotation in the CR3 frame, with no scale or reflection.
        half = np.sin(np.pi / 4.0)
        converted = quest_quaternion_to_robot_rotation(
            [0.0, half, 0.0, np.cos(np.pi / 4.0)]
        )
        np.testing.assert_allclose(converted.T @ converted, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(converted)), 1.0, places=12)

    def test_rotation_error_vector_is_bounded(self):
        angle = np.deg2rad(90.0)
        target = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        error = app.rotation_error_vector(target, np.eye(3), max_angle_rad=np.deg2rad(30.0))
        self.assertAlmostEqual(float(np.linalg.norm(error)), np.deg2rad(30.0), places=7)

    def test_quest_home_pose_locks_tool_z_to_world_x(self):
        # The third rotation column is the tool/TCP Z axis in the world frame.
        tool_z = app.QUEST_HOME_ROTATION[:, 2]
        self.assertLess(float(np.linalg.norm(tool_z - [1.0, 0.0, 0.0])), 0.06)
        np.testing.assert_allclose(
            app.QUEST_HOME_ROTATION.T @ app.QUEST_HOME_ROTATION,
            np.eye(3),
            atol=1e-12,
        )

    def test_quest_home_joints_are_inside_model_limits(self):
        model = mujoco.MjModel.from_xml_path(str(app.DEFAULT_MODEL))
        joint_ids, _, _ = app.joint_indices(model)
        limits = model.jnt_range[joint_ids]
        self.assertTrue(
            np.all(app.QUEST_HOME_Q_RAD >= limits[:, 0])
            and np.all(app.QUEST_HOME_Q_RAD <= limits[:, 1])
        )

    def test_quest_home_q_rad_for_mode_matches_selected_mode(self):
        np.testing.assert_allclose(
            app.quest_home_q_rad_for_mode(QUEST_MOTION_MODE_ORIGINAL),
            app.HOME_Q_RAD,
        )
        np.testing.assert_allclose(
            app.quest_home_q_rad_for_mode(QUEST_MOTION_MODE_REVERSED_END),
            app.QUEST_HOME_Q_RAD,
        )

    def test_cartesian_tracking_limit_uses_speed_and_caps_stale_dt(self):
        limited = gui.limit_cartesian_tracking_step(
            [0.10, 0.0, 0.0], max_speed_m_s=0.8, dt=0.02
        )
        self.assertAlmostEqual(np.linalg.norm(limited), 0.016)
        stale = gui.limit_cartesian_tracking_step(
            [0.10, 0.0, 0.0], max_speed_m_s=0.8, dt=1.0
        )
        self.assertAlmostEqual(
            np.linalg.norm(stale),
            0.8 * gui.QUEST_MAX_CONTROL_DT_S,
        )

    def test_quest_real_watchdog_stops_stale_wrist_stream(self):
        controller = object.__new__(gui.CR3ControlGUI)
        controller.quest_real_stage = "active"
        controller.quest_last_valid_wrist_at = 1.0
        controller.stop_quest_real_sync = Mock()
        controller._check_quest_real_sync(
            1.0 + gui.QUEST_REAL_HAND_WATCHDOG_S + 0.01
        )
        controller.stop_quest_real_sync.assert_called_once()
        self.assertIn(
            "腕部数据",
            controller.stop_quest_real_sync.call_args.args[0],
        )

    def test_quest_real_second_confirmation_expires(self):
        controller = object.__new__(gui.CR3ControlGUI)
        controller.quest_real_stage = "ready"
        controller.quest_real_confirm_until = 10.0
        controller.stop_quest_real_sync = Mock()
        controller._check_quest_real_sync(10.1)
        controller.stop_quest_real_sync.assert_called_once_with(
            "Quest 实机同步第二次确认已超时。"
        )

    def test_quest_real_speed_edit_applies_immediately(self):
        class Variable:
            def get(self):
                return "7.5"

        hardware = Mock()
        controller = object.__new__(gui.CR3ControlGUI)
        controller.live_hardware = hardware
        controller.quest_real_stage = "active"
        controller.quest_real_speed_var = Variable()
        controller.last_applied_live_speed = 5.0
        controller.log = Mock()
        controller._apply_quest_real_speed_setting()
        hardware.set_max_joint_speed.assert_called_once_with(7.5)
        self.assertEqual(controller.last_applied_live_speed, 7.5)

    def test_gui_language_translation_switches_both_directions(self):
        controller = object.__new__(gui.CR3ControlGUI)
        controller.language = "en"
        self.assertEqual(controller._tr("使能"), "Enable")
        self.assertEqual(controller._tr("控制器状态  ", "Controller status  "), "Controller status  ")
        controller.language = "zh"
        self.assertEqual(controller._tr("使能"), "使能")
        self.assertEqual(gui.UI_TEXT_ZH["Record"], "录制后回放")

    def test_gui_robot_ip_validation_and_pending_edit_lock(self):
        class Variable:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        controller = object.__new__(gui.CR3ControlGUI)
        controller.language = "zh"
        controller.robot_ip_var = Variable(" 192.168.5.12 ")
        controller.monitor = Mock(robot_ip="192.168.5.11")
        controller.connected_robot_ip = "192.168.5.11"
        controller.robot_connection_var = Mock()
        controller.log = Mock()

        self.assertEqual(
            controller._normalize_robot_ip(" 192.168.5.11 "),
            "192.168.5.11",
        )
        self.assertIsNone(controller._active_robot_ip())
        self.assertEqual(
            controller._connected_robot_ip_for_stop(),
            "192.168.5.11",
        )
        controller.log.assert_called()

        with self.assertRaises(ValueError):
            controller._normalize_robot_ip("192.168.5.999")

    @patch("run_keyboard_sim2real_gui.FeedbackReceiver")
    def test_gui_switches_feedback_and_commits_the_new_ip(self, receiver_type):
        class Variable:
            def __init__(self, value=""):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        old_receiver = Mock(robot_ip="192.168.5.11")
        new_receiver = Mock(robot_ip="192.168.5.12")
        state = Mock(
            joints_deg=np.zeros(6),
            robot_mode=5,
            enabled=True,
            error=False,
            paused=False,
            queue_running=False,
        )
        new_receiver.start.return_value = state
        receiver_type.return_value = new_receiver

        controller = object.__new__(gui.CR3ControlGUI)
        controller.language = "zh"
        controller.robot_ip_var = Variable("192.168.5.12")
        controller.robot_connection_var = Variable()
        controller.robot_status_var = Variable()
        controller.monitor = old_receiver
        controller.connected_robot_ip = "192.168.5.11"
        controller.feedback_connecting_ip = None
        controller.robot_ip_history = ["192.168.5.11"]
        controller.robot_ip_combo = Mock()
        controller.feedback_connect_button = Mock()
        controller.dashboard_action_busy = False
        controller.real_busy = False
        controller.live_hardware = None
        controller.live_starting = False
        controller.teach_active = False
        controller.teach_starting = False
        controller.quest_real_stage = "idle"
        controller.recorder = Mock(recording=False)
        controller.log = Mock()
        controller._update_feedback_display = Mock()
        controller._set_sim_from_real = Mock()
        controller._background = lambda work, success, _failure: success(work())

        controller.connect_feedback()

        old_receiver.close.assert_called_once()
        receiver_type.assert_called_once_with("192.168.5.12")
        self.assertIs(controller.monitor, new_receiver)
        self.assertEqual(controller.connected_robot_ip, "192.168.5.12")
        self.assertEqual(
            controller.robot_ip_history,
            ["192.168.5.11", "192.168.5.12"],
        )

    def test_english_runtime_log_translation_preserves_dynamic_values(self):
        self.assertEqual(
            gui.translate_runtime_log("平移步长 必须在 0.1 到 50 之间"),
            "translation step must be between 0.1 and 50",
        )
        quest_translated = gui.translate_runtime_log(
            "Quest UDP 正在监听 0.0.0.0:9000，控制手=right。"
        )
        self.assertEqual(
            quest_translated,
            "Quest UDP listening on 0.0.0.0:9000, control hand=right.",
        )
        self.assertFalse(any("\u3400" <= char <= "\u9fff" for char in quest_translated))
        self.assertEqual(gui.UI_TEXT_EN["清空日志"], "Clear Log")
        self.assertEqual(gui.UI_TEXT_EN["实机回 Home"], "Robot Home")

    def test_escape_forces_disable_without_requiring_canvas_focus(self):
        controller = object.__new__(gui.CR3ControlGUI)
        controller._clear_motion_keys = Mock()
        controller.on_disable = Mock()

        result = controller._key_press(Mock(keysym="Escape"))

        self.assertEqual(result, "break")
        controller._clear_motion_keys.assert_called_once_with()
        controller.on_disable.assert_called_once_with(confirm=False, force=True)

    @patch("run_keyboard_sim2real_gui.messagebox.askyesno")
    @patch("run_keyboard_sim2real_gui.disable_robot")
    def test_forced_disable_has_no_confirmation_dialog(self, disable, askyesno):
        disable.return_value = "0,{},DisableRobot();"
        controller = object.__new__(gui.CR3ControlGUI)
        controller.args = Mock(enable_real_execution=True)
        controller.disable_in_progress = False
        controller.dashboard_action_busy = True
        controller._connected_robot_ip_for_stop = Mock(
            return_value="192.168.5.11"
        )
        controller.real_busy = True
        controller.recorder = Mock(recording=False)
        controller._clear_motion_keys = Mock()
        controller.real_stop_event = Mock()
        controller.arm_status_var = Mock()
        controller.live_starting = True
        controller.quest_real_stage = "active"
        controller.quest_real_confirm_until = 10.0
        controller.teach_active = True
        controller.teach_starting = True
        controller.teach_button = Mock()
        controller.teach_status_var = Mock()
        controller.live_hardware = None
        controller.last_applied_live_speed = 5.0
        controller.live_button = Mock()
        controller.live_speed_status_var = Mock()
        controller.status_var = Mock()
        controller.monitor = Mock()
        controller.log = Mock()
        controller._set_button_text = Mock()
        controller._background = lambda work, success, _failure: success(work())

        controller.on_disable(confirm=False, force=True)

        askyesno.assert_not_called()
        disable.assert_called_once_with(
            "192.168.5.11", feedback=controller.monitor
        )
        controller.real_stop_event.set.assert_called_once_with()
        controller._clear_motion_keys.assert_called_once_with()
        self.assertIsNone(controller.live_hardware)
        self.assertEqual(controller.quest_real_stage, "idle")
        self.assertFalse(controller.teach_active)
        self.assertFalse(controller.disable_in_progress)

    def test_manual_real_home_completion_realigns_hand_origins(self):
        controller = object.__new__(gui.CR3ControlGUI)
        controller.real_busy = True
        controller.manual_home_active = True
        controller.real_home_button = Mock()
        controller._set_button_text = Mock()
        home_ee = np.array([0.1, 0.2, 0.3])
        controller._set_sim_home_now = Mock(return_value=home_ee)
        controller.quest_mapper = Mock(calibrated=True)
        controller._update_feedback_display = Mock()
        controller.status_var = Mock()
        controller.log = Mock()
        state = Mock()

        controller._real_home_complete(state)

        self.assertFalse(controller.real_busy)
        self.assertFalse(controller.manual_home_active)
        controller.quest_mapper.reanchor_robot_origin.assert_called_once_with(home_ee)
        controller._update_feedback_display.assert_called_once_with(state)

    def test_manual_real_home_button_can_cancel_homing(self):
        controller = object.__new__(gui.CR3ControlGUI)
        controller.manual_home_active = True
        controller.real_stop_event = Mock()
        controller.real_home_button = Mock()
        controller._set_button_text = Mock()
        controller.log = Mock()

        controller.toggle_real_home()

        controller.real_stop_event.set.assert_called_once_with()
        controller._set_button_text.assert_called_once_with(
            controller.real_home_button, "正在停止实机 Home…"
        )

    @patch("cr3_sim2real.hardware.socket.create_connection")
    def test_feedback_receiver_reconnects_after_connection_reset(
        self, create_connection
    ):
        packet = np.zeros(1, dtype=MyType)
        packet["test_value"][0] = PACKET_MARKER
        packet["robot_mode"][0] = 5
        packet["enable_status"][0] = 1
        packet["q_actual"][0] = np.arange(6, dtype=float)

        class FakeSocket:
            def __init__(self, responses):
                self.responses = list(responses)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def settimeout(self, _timeout):
                pass

            def recv(self, _size):
                if self.responses:
                    response = self.responses.pop(0)
                    if isinstance(response, Exception):
                        raise response
                    return response
                raise socket.timeout()

            def shutdown(self, _how):
                pass

            def close(self):
                pass

        create_connection.side_effect = [
            FakeSocket([ConnectionResetError(10054, "reset")]),
            FakeSocket([packet.tobytes()]),
        ]
        receiver = FeedbackReceiver("192.0.2.1")
        try:
            state = receiver.start(timeout=1.5)
            np.testing.assert_allclose(state.joints_deg, np.arange(6, dtype=float))
            self.assertTrue(state.enabled)
            self.assertEqual(create_connection.call_count, 2)
        finally:
            receiver.close()

    def test_hamer_cam_translation_validation(self):
        np.testing.assert_allclose(
            hamer_cam_t({"cam_t": [0.1, -0.2, 1.4]}),
            [0.1, -0.2, 1.4],
        )
        self.assertIsNone(hamer_cam_t(None))
        self.assertIsNone(hamer_cam_t({"cam_t": [1.0, 2.0]}))
        self.assertIsNone(hamer_cam_t({"cam_t": [1.0, np.nan, 2.0]}))

    def test_hamer_mapper_requires_manual_origin(self):
        mapper = HamerArmMapper(
            depth_scale=1.0,
            lateral_vertical_scale=1.0,
            depth_deadband=0.0,
            max_delta=1.0,
            ema_alpha=1.0,
            filter_deadzone=0.0,
            filter_max_step=None,
        )
        with self.assertRaisesRegex(RuntimeError, "origin"):
            mapper.target_pos(np.array([0.0, 0.0, 1.0]))
        mapper.calibrate(
            np.array([0.0, 0.0, 1.0]),
            np.array([0.4, -0.1, 0.5]),
        )
        np.testing.assert_allclose(
            mapper.target_pos(np.array([0.1, -0.2, 1.3])),
            [0.1, 0.0, 0.7],
        )
        hand_origin = mapper.cam_origin.copy()
        mapper.reanchor_robot_origin(np.array([0.2, 0.3, 0.4]))
        np.testing.assert_allclose(mapper.cam_origin, hand_origin)
        np.testing.assert_allclose(
            mapper.target_pos(hand_origin),
            [0.2, 0.3, 0.4],
        )
        mapper.clear_origin()
        self.assertFalse(mapper.calibrated)

    @patch("cr3_sim2real.hamer_bridge.urllib.request.urlopen")
    @patch("cr3_sim2real.hamer_bridge.cv2.VideoCapture")
    def test_hamer_camera_keeps_updating_during_slow_original_inference(
        self,
        video_capture,
        urlopen,
    ):
        class FakeCapture:
            def __init__(self):
                self.released = False
                self.index = 0

            def isOpened(self):
                return True

            def set(self, *_args):
                return True

            def read(self):
                time.sleep(0.005)
                if self.released:
                    return False, None
                self.index += 1
                return True, np.full((48, 64, 3), self.index % 255, dtype=np.uint8)

            def release(self):
                self.released = True

        class SlowResponse:
            def __enter__(self):
                time.sleep(0.10)
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"ok":true,"status":"HaMeR right hand detected","result":{"cam_t":[0,0,1]}}'

        video_capture.return_value = FakeCapture()
        urlopen.side_effect = lambda *_args, **_kwargs: SlowResponse()
        tracker = HamerBridgeTracker("http://127.0.0.1:8765", camera_backend="any")
        tracker.start()
        try:
            deadline = time.monotonic() + 1.0
            snapshot = tracker.get()
            while snapshot.sequence < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
                snapshot = tracker.get()
            self.assertGreaterEqual(snapshot.sequence, 1)
            self.assertGreater(snapshot.frame_sequence, 10)
            np.testing.assert_allclose(hamer_cam_t(snapshot.result), [0, 0, 1])
        finally:
            tracker.stop()

    def test_only_numpad_keys_are_accepted(self):
        self.assertTrue(app.ACCEPTED_NUMPAD_KEYS)
        self.assertNotIn(glfw.KEY_8, app.ACCEPTED_NUMPAD_KEYS)
        self.assertNotIn(glfw.KEY_R, app.ACCEPTED_NUMPAD_KEYS)
        self.assertTrue(
            all(glfw.KEY_KP_0 <= key <= glfw.KEY_KP_EQUAL for key in app.ACCEPTED_NUMPAD_KEYS)
        )

    def test_gui_numpad_twist_uses_adjustable_steps(self):
        translation = gui.numpad_twist(
            "KP_9", translation_step_m=0.012, rotation_step_rad=0.03
        )
        rotation = gui.numpad_twist(
            "KP_Add", translation_step_m=0.012, rotation_step_rad=0.03
        )
        np.testing.assert_allclose(translation, [0.012, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(rotation, [0, 0, 0, 0, 0, 0.03])
        self.assertIsNone(
            gui.numpad_twist(
                "9", translation_step_m=0.012, rotation_step_rad=0.03
            )
        )

    def test_gui_full_keyboard_controls_translation_and_rotation(self):
        translation = gui.keyboard_twist(
            "w", translation_step_m=0.01, rotation_step_rad=0.04
        )
        rotation = gui.keyboard_twist(
            "L", translation_step_m=0.01, rotation_step_rad=0.04
        )
        main_number = gui.keyboard_twist(
            "9", translation_step_m=0.01, rotation_step_rad=0.04
        )
        np.testing.assert_allclose(translation, [0, 0, 0.01, 0, 0, 0])
        np.testing.assert_allclose(rotation, [0, 0, 0, 0, 0.04, 0])
        np.testing.assert_allclose(main_number, [0.01, 0, 0, 0, 0, 0])

    def test_gui_renderer_tracks_canvas_resolution(self):
        self.assertEqual(gui.render_size_for_canvas(1280, 700), (1280, 700))
        self.assertEqual(
            gui.render_size_for_canvas(100, 100),
            (gui.MIN_RENDER_WIDTH, gui.MIN_RENDER_HEIGHT),
        )

    def test_mapping_round_trip(self):
        real = np.array([1.0, 25.0, -65.0, 28.0, 5.0, 4.0])
        np.testing.assert_allclose(
            sim_rad_to_real_deg(real_deg_to_sim_rad(real)), real, atol=1e-12
        )

    def test_real_speed_must_be_below_twenty(self):
        self.assertEqual(app.percent_below_twenty("19"), 19)
        with self.assertRaises(Exception):
            app.percent_below_twenty("20")
        self.assertEqual(app.positive_float("2"), 2.0)
        with self.assertRaises(Exception):
            app.positive_float("0")
        self.assertEqual(
            app.live_speed_below_twenty(str(DEFAULT_LIVE_JOINT_SPEED_DEG_S)),
            DEFAULT_LIVE_JOINT_SPEED_DEG_S,
        )
        with self.assertRaises(Exception):
            app.live_speed_below_twenty(str(STRICT_20_PERCENT_LIMIT_DEG_S))

    def test_live_servo_rate_limit_is_strictly_below_twenty_percent(self):
        limited = limit_joint_velocity(
            np.zeros(6),
            np.full(6, 100.0),
            max_joint_speed_deg_s=18.0,
            dt=0.03,
        )
        np.testing.assert_allclose(limited, np.full(6, 0.54))
        with self.assertRaises(ValueError):
            limit_joint_velocity(
                np.zeros(6),
                np.ones(6),
                max_joint_speed_deg_s=36.0,
                dt=0.03,
            )

    def test_live_servo_runtime_speed_change_is_immediate(self):
        ready = RobotFeedback(
            received_at=0.0,
            joints_deg=np.zeros(6),
            joint_speeds_deg_s=np.zeros(6),
            robot_mode=5,
            enabled=True,
            error=False,
        )

        class Feedback:
            def require_fresh(self):
                return ready

            def close(self):
                pass

        class Move:
            def __init__(self):
                self.commands = []

            def ServoJ(self, *joints, **parameters):
                self.commands.append((np.asarray(joints), parameters))
                return "0,{},ServoJ(...);"

            def close(self):
                pass

        hardware = LiveServoHardware("192.0.2.1", max_joint_speed_deg_s=18.0)
        hardware.feedback = Feedback()
        hardware.move = Move()
        hardware._last_command_deg = np.zeros(6)
        hardware._last_send_time = 0.0

        first = hardware.send_if_due(np.full(6, 100.0), now=10.0)
        np.testing.assert_allclose(
            first, np.full(6, 18.0 * LIVE_SERVO_PERIOD_S)
        )
        self.assertAlmostEqual(hardware.last_planned_speed_deg_s, 18.0)

        hardware.set_max_joint_speed(2.0)
        second = hardware.send_if_due(
            np.full(6, 100.0), now=10.0 + LIVE_SERVO_PERIOD_S + 1e-6
        )
        np.testing.assert_allclose(
            second, first + np.full(6, 2.0 * LIVE_SERVO_PERIOD_S)
        )
        self.assertAlmostEqual(hardware.last_planned_speed_deg_s, 2.0)
        with self.assertRaises(ValueError):
            hardware.set_max_joint_speed(STRICT_20_PERCENT_LIMIT_DEG_S)

    def test_live_servo_stream_runs_independently_at_fixed_rate(self):
        ready = RobotFeedback(
            received_at=time.monotonic(),
            joints_deg=np.zeros(6),
            joint_speeds_deg_s=np.zeros(6),
            robot_mode=5,
            enabled=True,
            error=False,
        )

        class Feedback:
            def require_fresh(self):
                return ready

            def close(self):
                pass

        class Move:
            def __init__(self):
                self.command_times = []

            def ServoJ(self, *joints, **parameters):
                self.command_times.append(time.monotonic())
                return "0,{},ServoJ(...);"

            def close(self):
                pass

        hardware = LiveServoHardware("192.0.2.1", max_joint_speed_deg_s=3.0)
        hardware.feedback = Feedback()
        hardware.move = Move()
        hardware._last_command_deg = np.zeros(6)
        hardware._last_send_time = time.monotonic()
        hardware.start_stream(np.full(6, 20.0))
        deadline = time.monotonic() + 0.25
        while len(hardware.move.command_times) < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        command_times = hardware.move.command_times.copy()
        hardware.close()

        self.assertGreaterEqual(len(command_times), 3)
        intervals = np.diff(command_times[:3])
        self.assertTrue(np.all(intervals >= LIVE_SERVO_PERIOD_S * 0.75))
        self.assertTrue(np.all(intervals <= LIVE_SERVO_PERIOD_S * 1.75))

    def test_live_servo_smooths_discrete_joint_targets_before_limiting(self):
        ready = RobotFeedback(
            received_at=0.0,
            joints_deg=np.zeros(6),
            joint_speeds_deg_s=np.zeros(6),
            robot_mode=5,
            enabled=True,
            error=False,
        )

        class Feedback:
            def require_fresh(self):
                return ready

        class Move:
            def ServoJ(self, *joints, **parameters):
                return "0,{},ServoJ(...);"

        hardware = LiveServoHardware(
            "192.0.2.1",
            max_joint_speed_deg_s=18.0,
            max_tracking_error_deg=100.0,
        )
        hardware.feedback = Feedback()
        hardware.move = Move()
        hardware._last_command_deg = np.zeros(6)
        hardware._filtered_target_deg = np.zeros(6)
        hardware._previous_requested_deg = np.zeros(6)
        hardware._last_send_time = 0.0

        requested = np.full(6, 0.015)
        first = hardware.send_if_due(requested, now=10.0)
        np.testing.assert_allclose(
            first,
            requested * LIVE_TARGET_FILTER_SLOW_ALPHA,
        )
        self.assertAlmostEqual(
            hardware.last_target_filter_alpha,
            LIVE_TARGET_FILTER_SLOW_ALPHA,
        )
        second_request = np.full(6, 0.60)
        second = hardware.send_if_due(
            second_request,
            now=10.0 + LIVE_SERVO_PERIOD_S + 1e-6,
        )
        self.assertTrue(np.all(second > first))
        self.assertTrue(np.all(second < second_request))
        self.assertAlmostEqual(
            hardware.last_target_filter_alpha,
            LIVE_TARGET_FILTER_FAST_ALPHA,
        )
        np.testing.assert_allclose(hardware.latest_command_deg(), second)

    @patch("cr3_sim2real.hardware.DobotApiMove")
    @patch("cr3_sim2real.hardware.DobotApiDashboard")
    def test_live_servo_reuses_gui_feedback_without_closing_it(
        self, dashboard_class, move_class
    ):
        ready = RobotFeedback(
            received_at=0.0,
            joints_deg=np.zeros(6),
            joint_speeds_deg_s=np.zeros(6),
            robot_mode=5,
            enabled=True,
            error=False,
            queue_running=True,
        )

        class SharedFeedback:
            def __init__(self):
                self.start_calls = 0
                self.close_calls = 0

            def start(self):
                self.start_calls += 1
                return ready

            def require_fresh(self, max_age=0.2):
                return ready

            def close(self):
                self.close_calls += 1

        shared = SharedFeedback()
        hardware = LiveServoHardware(
            "192.0.2.1",
            max_joint_speed_deg_s=3.0,
            feedback=shared,
        )
        state = hardware.connect()
        hardware.close()

        self.assertIs(state, ready)
        self.assertEqual(shared.start_calls, 0)
        self.assertEqual(shared.close_calls, 0)
        dashboard_class.return_value.close.assert_called_once_with()
        move_class.return_value.close.assert_called_once_with()

    @patch("cr3_sim2real.hardware.DobotApiMove")
    @patch("cr3_sim2real.hardware.DobotApiDashboard")
    def test_playback_reuses_gui_feedback_without_closing_it(
        self, dashboard_class, move_class
    ):
        ready = RobotFeedback(
            received_at=0.0,
            joints_deg=np.zeros(6),
            joint_speeds_deg_s=np.zeros(6),
            robot_mode=5,
            enabled=True,
            error=False,
            queue_running=True,
        )

        class SharedFeedback:
            def __init__(self):
                self.start_calls = 0
                self.close_calls = 0

            def start(self):
                self.start_calls += 1
                return ready

            def require_fresh(self, max_age=0.2):
                return ready

            def close(self):
                self.close_calls += 1

        shared = SharedFeedback()
        hardware = PlaybackHardware(
            "192.0.2.1", 10, 5, feedback=shared
        )
        state = hardware.connect()
        hardware.close()

        self.assertIs(state, ready)
        self.assertEqual(shared.start_calls, 0)
        self.assertEqual(shared.close_calls, 0)
        dashboard_class.return_value.close.assert_called_once_with()
        move_class.return_value.close.assert_called_once_with()

    def test_live_servo_idle_time_cannot_bypass_speed_limit(self):
        ready = RobotFeedback(
            received_at=0.0,
            joints_deg=np.zeros(6),
            joint_speeds_deg_s=np.zeros(6),
            robot_mode=5,
            enabled=True,
            error=False,
        )

        class Feedback:
            def require_fresh(self):
                return ready

            def close(self):
                pass

        class Move:
            def ServoJ(self, *joints, **parameters):
                return "0,{},ServoJ(...);"

            def close(self):
                pass

        hardware = LiveServoHardware("192.0.2.1", max_joint_speed_deg_s=3.0)
        hardware.feedback = Feedback()
        hardware.move = Move()
        hardware._last_command_deg = np.zeros(6)
        hardware._last_send_time = 0.0

        idle = hardware.send_if_due(np.zeros(6), now=100.0)
        np.testing.assert_allclose(idle, np.zeros(6))
        self.assertEqual(hardware._last_send_time, 100.0)

        moved = hardware.send_if_due(
            np.full(6, 100.0), now=100.0 + LIVE_SERVO_PERIOD_S
        )
        np.testing.assert_allclose(
            moved, np.full(6, 3.0 * LIVE_SERVO_PERIOD_S)
        )

    def test_live_servo_stops_on_excessive_tracking_error(self):
        lagging = RobotFeedback(
            received_at=0.0,
            joints_deg=np.zeros(6),
            joint_speeds_deg_s=np.zeros(6),
            robot_mode=5,
            enabled=True,
            error=False,
        )

        class Feedback:
            def require_fresh(self):
                return lagging

        hardware = LiveServoHardware(
            "192.0.2.1",
            max_joint_speed_deg_s=3.0,
            max_tracking_error_deg=5.0,
        )
        hardware.feedback = Feedback()
        hardware.move = object()
        hardware._last_command_deg = np.full(6, 6.0)
        hardware._last_send_time = 0.0
        with self.assertRaisesRegex(RuntimeError, "tracking error"):
            hardware.send_if_due(np.full(6, 7.0), now=1.0)

    def test_motion_ready_requires_enabled_error_free_robot(self):
        good = RobotFeedback(
            received_at=0.0,
            joints_deg=np.zeros(6),
            joint_speeds_deg_s=np.zeros(6),
            robot_mode=5,
            enabled=True,
            error=False,
        )
        require_motion_ready(good)
        with self.assertRaises(RuntimeError):
            require_motion_ready(
                RobotFeedback(
                    received_at=0.0,
                    joints_deg=np.zeros(6),
                    joint_speeds_deg_s=np.zeros(6),
                    robot_mode=9,
                    enabled=True,
                    error=True,
                )
            )

    def test_nonzero_dobot_reply_stops_execution(self):
        require_command_success("JointMovJ", "0,{},JointMovJ(...);")
        require_command_success("ClearError", b"0,{},ClearError();")
        with self.assertRaisesRegex(RuntimeError, "not tcp mode"):
            require_command_success(
                "JointMovJ",
                "-2,{not tcp mode or system is starting},JointMovJ(...);",
            )
        with self.assertRaisesRegex(RuntimeError, "unrecognized reply"):
            require_command_success("Sync", "")

    @patch("cr3_sim2real.hardware.DobotApiDashboard")
    def test_dashboard_recovery_commands_are_explicit_and_checked(self, dashboard_class):
        dashboard = dashboard_class.return_value
        dashboard.ClearError.return_value = "0,{},ClearError();"
        dashboard.pause.return_value = "0,{},pause();"
        dashboard.Continue.return_value = "0,{},continue();"

        self.assertIn("ClearError", clear_robot_error("192.0.2.1"))
        self.assertIn("pause", pause_robot_queue("192.0.2.1"))
        self.assertIn("continue", continue_robot_queue("192.0.2.1"))
        dashboard.ClearError.assert_called_once_with()
        dashboard.pause.assert_called_once_with()
        dashboard.Continue.assert_called_once_with()
        self.assertEqual(dashboard.close.call_count, 3)

    @patch("cr3_sim2real.hardware.DobotApiDashboard")
    def test_global_speed_factor_uses_strict_safety_cap(self, dashboard_class):
        dashboard = dashboard_class.return_value
        dashboard.SpeedFactor.return_value = "0,{},SpeedFactor(10);"
        self.assertIn("SpeedFactor", set_robot_speed_factor("192.0.2.1", 10))
        dashboard.SpeedFactor.assert_called_once_with(10)
        dashboard.close.assert_called_once_with()
        for invalid in (0, 20, 100, 2.5, True):
            with self.assertRaises(ValueError):
                set_robot_speed_factor("192.0.2.1", invalid)

    @patch("cr3_sim2real.hardware.DobotApiDashboard")
    def test_teaching_drag_commands_use_dashboard_api(self, dashboard_class):
        dashboard = dashboard_class.return_value
        dashboard.StartDrag.return_value = "0,{},StartDrag();"
        dashboard.StopDrag.return_value = "0,{},StopDrag();"

        self.assertIn("StartDrag", start_drag_mode("192.0.2.1"))
        self.assertIn("StopDrag", stop_drag_mode("192.0.2.1"))
        dashboard.StartDrag.assert_called_once_with()
        dashboard.StopDrag.assert_called_once_with()
        self.assertEqual(dashboard.close.call_count, 2)

    def test_drag_mode_is_verified_from_feedback(self):
        idle = RobotFeedback(
            received_at=0.0,
            joints_deg=np.zeros(6),
            joint_speeds_deg_s=np.zeros(6),
            robot_mode=5,
            enabled=True,
            error=False,
            drag_status=False,
        )
        dragging = RobotFeedback(
            received_at=0.0,
            joints_deg=np.zeros(6),
            joint_speeds_deg_s=np.zeros(6),
            robot_mode=6,
            enabled=True,
            error=False,
            drag_status=True,
        )

        class Feedback:
            def __init__(self, states):
                self.states = list(states)

            def require_fresh(self, max_age=0.5):
                if len(self.states) > 1:
                    return self.states.pop(0)
                return self.states[0]

        entered = wait_for_drag_state(Feedback([idle, dragging]), True, timeout=0.1)
        self.assertEqual(entered.robot_mode, 6)
        exited = wait_for_drag_state(Feedback([dragging, idle]), False, timeout=0.1)
        self.assertEqual(exited.robot_mode, 5)
        with self.assertRaises(TimeoutError):
            wait_for_drag_state(Feedback([idle]), True, timeout=0.01)

    @patch("cr3_sim2real.hardware.DobotApiDashboard")
    def test_get_error_ids_parses_controller_and_servo_groups(self, dashboard_class):
        dashboard_class.return_value.GetErrorID.return_value = (
            "0,{[[16],[2],[],[],[],[],[]]},GetErrorID();"
        )
        groups, reply = get_robot_error_ids("192.0.2.1")
        self.assertEqual(groups, [[16], [2], [], [], [], [], []])
        self.assertIn("GetErrorID", reply)
        descriptions = gui.describe_alarm_groups(groups)
        self.assertTrue(any("控制器 [16]" in item for item in descriptions))
        self.assertTrue(any("伺服 J1 [2]" in item for item in descriptions))
        english = gui.describe_alarm_groups(groups, "en")
        self.assertTrue(any("Controller [16]" in item for item in english))
        self.assertTrue(any("Servo J1 [2]" in item for item in english))
        self.assertFalse(
            any("\u3400" <= char <= "\u9fff" for item in english for char in item)
        )

    @patch("cr3_sim2real.hardware.DobotApiDashboard")
    def test_get_error_ids_rejects_malformed_payload(self, dashboard_class):
        dashboard_class.return_value.GetErrorID.return_value = (
            "0,{not-json},GetErrorID();"
        )
        with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
            get_robot_error_ids("192.0.2.1")

    @patch("cr3_sim2real.hardware.DobotApiDashboard")
    def test_power_on_and_disable_use_dashboard_commands(self, dashboard_class):
        dashboard = dashboard_class.return_value
        dashboard.PowerOn.return_value = "0,{},PowerOn();"
        dashboard.DisableRobot.return_value = "0,{},DisableRobot();"

        self.assertIn("PowerOn", power_on_robot("192.0.2.1"))
        self.assertIn("DisableRobot", disable_robot("192.0.2.1"))

        dashboard.PowerOn.assert_called_once_with()
        dashboard.DisableRobot.assert_called_once_with()
        self.assertEqual(dashboard.close.call_count, 2)

    @patch("cr3_sim2real.hardware.DobotApiDashboard")
    def test_disable_empty_reply_is_verified_by_feedback(self, dashboard_class):
        dashboard_class.return_value.DisableRobot.return_value = b""
        disabled = RobotFeedback(
            received_at=0.0,
            joints_deg=np.zeros(6),
            joint_speeds_deg_s=np.zeros(6),
            robot_mode=4,
            enabled=False,
            error=False,
        )

        class Feedback:
            def require_fresh(self, max_age=0.5):
                return disabled

        reply = disable_robot("192.0.2.1", feedback=Feedback())
        self.assertIn("verified enable_status=0", reply)

    def test_playback_streams_servoj_and_only_waits_at_final_target(self):
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        arrived = RobotFeedback(
            received_at=0.0,
            joints_deg=target.copy(),
            joint_speeds_deg_s=np.zeros(6),
            robot_mode=5,
            enabled=True,
            error=False,
        )

        class StaticFeedback:
            queue_running = False

            def require_fresh(self):
                return RobotFeedback(
                    received_at=arrived.received_at,
                    joints_deg=arrived.joints_deg,
                    joint_speeds_deg_s=arrived.joint_speeds_deg_s,
                    robot_mode=arrived.robot_mode,
                    enabled=arrived.enabled,
                    error=arrived.error,
                    queue_running=self.queue_running,
                )

        class Dashboard:
            def __init__(self, feedback):
                self.feedback = feedback
                self.continue_calls = 0

            def Continue(self):
                self.continue_calls += 1
                self.feedback.queue_running = True
                return "0,{},continue();"

        class MoveWithoutSync:
            def __init__(self):
                self.commands = []

            def ServoJ(self, *args, **kwargs):
                self.commands.append((args, kwargs))
                return "0,{},ServoJ(...);"

        hardware = PlaybackHardware("192.0.2.1", 10, 5)
        hardware.feedback = StaticFeedback()
        hardware.dashboard = Dashboard(hardware.feedback)
        hardware.move = MoveWithoutSync()
        hardware._last_command_deg = target.copy()
        hardware.execute(
            [
                (0.0, target),
                (0.001, target),
                (0.002, target),
            ]
        )
        self.assertEqual(len(hardware.move.commands), 3)
        self.assertEqual(hardware.dashboard.continue_calls, 1)

    def test_joint_target_requires_position_and_stopped_speed(self):
        target = np.zeros(6)
        feedback = RobotFeedback(
            received_at=0.0,
            joints_deg=np.full(6, 0.1),
            joint_speeds_deg_s=np.full(6, 0.1),
            robot_mode=5,
            enabled=True,
            error=False,
        )
        self.assertTrue(joint_target_reached(feedback, target))
        self.assertFalse(
            joint_target_reached(
                RobotFeedback(
                    received_at=0.0,
                    joints_deg=np.full(6, 0.3),
                    joint_speeds_deg_s=np.zeros(6),
                    robot_mode=5,
                    enabled=True,
                    error=False,
                ),
                target,
            )
        )

    def test_live_home_stream_uses_limiter_until_feedback_reaches_target(self):
        target = np.ones(6)

        class Feedback:
            def __init__(self):
                self.index = 0

            def require_fresh(self):
                self.index += 1
                reached = self.index >= 3
                return RobotFeedback(
                    received_at=time.monotonic(),
                    joints_deg=target.copy() if reached else np.zeros(6),
                    joint_speeds_deg_s=np.zeros(6),
                    robot_mode=5,
                    enabled=True,
                    error=False,
                )

        class Hardware:
            def __init__(self):
                self.feedback = Feedback()
                self.targets = []

            def send_if_due(self, requested):
                self.targets.append(np.asarray(requested).copy())
                return self.targets[-1]

        hardware = Hardware()
        reached = stream_live_target_until_reached(hardware, target, timeout=1.0)
        np.testing.assert_allclose(reached.joints_deg, target)
        self.assertEqual(len(hardware.targets), 2)
        np.testing.assert_allclose(hardware.targets[0], target)
        self.assertFalse(
            joint_target_reached(
                RobotFeedback(
                    received_at=0.0,
                    joints_deg=np.zeros(6),
                    joint_speeds_deg_s=np.ones(6),
                    robot_mode=7,
                    enabled=True,
                    error=False,
                ),
                target,
            )
        )

    def test_cartesian_ik_changes_six_joint_target_within_limits(self):
        model = mujoco.MjModel.from_xml_path(str(app.DEFAULT_MODEL))
        data = mujoco.MjData(model)
        joint_ids, qpos_indices, dof_indices = app.joint_indices(model)
        data.qpos[qpos_indices] = app.HOME_Q_RAD
        mujoco.mj_forward(model, data)
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link6")
        result = app.apply_cartesian_increment(
            model,
            data,
            ee_id,
            dof_indices,
            joint_ids,
            app.HOME_Q_RAD.copy(),
            app.NUMPAD_TWIST[glfw.KEY_KP_9],
        )
        self.assertEqual(result.shape, (6,))
        self.assertGreater(np.linalg.norm(result - app.HOME_Q_RAD), 0.0)
        self.assertTrue(np.all(result >= model.jnt_range[joint_ids, 0]))
        self.assertTrue(np.all(result <= model.jnt_range[joint_ids, 1]))

    def test_default_model_has_quest_mocap_marker(self):
        model = mujoco.MjModel.from_xml_path(str(app.DEFAULT_MODEL))
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quest_wrist_marker")
        self.assertGreaterEqual(body_id, 0)
        self.assertGreaterEqual(model.nmocap, 1)
        self.assertEqual(model.body_mocapid[body_id], 0)
        geoms = np.flatnonzero(model.geom_bodyid == body_id)
        self.assertEqual(geoms.size, 1)
        self.assertGreater(model.geom_rgba[geoms[0], 3], 0.0)

    def test_record_save_validate_and_dry_run(self):
        recorder = TrajectoryRecorder(sample_period=0.02)
        q0 = app.HOME_Q_RAD.copy()
        q1 = q0 + np.deg2rad([0.1, 0, 0, 0, 0, 0])
        recorder.start(10.0, q0, [0.1, 0.2, 0.3])
        recorder.sample(10.02, q1, [0.101, 0.2, 0.3])
        recorder.stop(10.04, q1, [0.101, 0.2, 0.3])
        self.assertTrue(recorder.review_ready)

        model = mujoco.MjModel.from_xml_path(str(app.DEFAULT_MODEL))
        joint_ids, _, _ = app.joint_indices(model)
        self.assertEqual(
            validate_trajectory(recorder.points, model.jnt_range[joint_ids]), []
        )
        commands = build_jointmovj_dry_run(recorder.points)
        self.assertTrue(commands)
        self.assertTrue(all("SpeedJ=10" in command for command in commands))
        self.assertTrue(all("AccJ=10" in command for command in commands))
        servo_commands = build_servoj_dry_run(recorder.points, sample_period=0.03)
        self.assertEqual(len(servo_commands), 3)
        self.assertTrue(all("ServoJ(" in command for command in servo_commands))

        with tempfile.TemporaryDirectory() as directory:
            path = recorder.save_json(Path(directory))
            self.assertTrue(path.is_file())

    def test_load_json_round_trip(self):
        recorder = TrajectoryRecorder(sample_period=0.02)
        q0 = app.HOME_Q_RAD.copy()
        q1 = q0 + np.deg2rad([0.1, 0, 0, 0, 0, 0])
        recorder.start(10.0, q0, [0.1, 0.2, 0.3])
        recorder.sample(10.02, q1, [0.101, 0.2, 0.3])
        recorder.stop(10.04, q1, [0.101, 0.2, 0.3])
        with tempfile.TemporaryDirectory() as directory:
            path = recorder.save_to(Path(directory) / "traj_a.json")
            loaded = TrajectoryRecorder(sample_period=0.02)
            count = loaded.load_json(path)
            self.assertEqual(count, len(recorder.points))
            self.assertTrue(loaded.review_ready)
            self.assertFalse(loaded.recording)
            np.testing.assert_allclose(
                [p.q for p in loaded.points], [p.q for p in recorder.points]
            )

    def test_load_json_rejects_wrong_format(self):
        recorder = TrajectoryRecorder()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"format": "other", "points": []}', encoding="utf-8")
            with self.assertRaises(ValueError):
                recorder.load_json(path)

    def test_save_to_overwrites_same_named_path(self):
        recorder = TrajectoryRecorder(sample_period=0.02)
        q0 = app.HOME_Q_RAD.copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "same.json"
            recorder.start(10.0, q0, [0.1, 0.2, 0.3])
            recorder.stop(
                10.02, q0 + np.deg2rad([0.2, 0, 0, 0, 0, 0]), [0.1, 0.2, 0.3]
            )
            recorder.save_to(path)
            loaded_first = TrajectoryRecorder()
            loaded_first.load_json(path)
            recorder.start(20.0, q0, [0.1, 0.2, 0.3])
            recorder.stop(
                20.02, q0 + np.deg2rad([0.5, 0, 0, 0, 0, 0]), [0.1, 0.2, 0.3]
            )
            recorder.save_to(path)
            loaded_second = TrajectoryRecorder()
            loaded_second.load_json(path)
            self.assertEqual(
                loaded_second.points[-1].q[0], q0[0] + np.deg2rad(0.5)
            )
            self.assertNotEqual(
                loaded_first.points[-1].q[0],
                loaded_second.points[-1].q[0],
            )

    def test_downsample_removes_stationary_duplicate_waypoints(self):
        q0 = np.zeros(6)
        q1 = np.deg2rad([1.0, 0, 0, 0, 0, 0])
        points = [
            TrajectoryPoint(0.0, q0, np.zeros(3)),
            TrajectoryPoint(0.5, q0, np.zeros(3)),
            TrajectoryPoint(1.0, q0, np.zeros(3)),
            TrajectoryPoint(1.5, q1, np.zeros(3)),
        ]
        selected = downsample_points(points)
        self.assertEqual(len(selected), 2)
        np.testing.assert_allclose(selected[0].q, q0)
        np.testing.assert_allclose(selected[1].q, q1)

    def test_servoj_resampling_preserves_timing_and_interpolates(self):
        points = [
            TrajectoryPoint(3.0, np.zeros(6), np.zeros(3)),
            TrajectoryPoint(3.1, np.ones(6), np.ones(3)),
        ]
        samples = resample_joint_trajectory(points, sample_period=0.03)
        np.testing.assert_allclose(
            [point.time for point in samples],
            [0.0, 0.03, 0.06, 0.09, 0.10],
        )
        np.testing.assert_allclose(samples[1].q, np.full(6, 0.3))
        np.testing.assert_allclose(samples[-1].q, np.ones(6))

    def test_hamer_real_prototype_only_plans_rate_limited_commands(self):
        feedback = RobotFeedback(
            received_at=0.0,
            joints_deg=np.zeros(6),
            joint_speeds_deg_s=np.zeros(6),
            robot_mode=5,
            enabled=True,
            error=False,
        )
        preview = HamerRealCommandPreview(
            HamerRealSafetyConfig(
                max_joint_speed_deg_s=5.0,
                hand_watchdog_s=0.5,
            )
        )
        self.assertEqual(preview.arm(np.zeros(6), feedback, now=1.0), 0.0)
        plan = preview.plan(
            np.deg2rad(np.full(6, 10.0)),
            feedback,
            now=1.1,
        )
        np.testing.assert_allclose(plan.requested_deg, np.full(6, 10.0))
        np.testing.assert_allclose(plan.planned_deg, np.full(6, 0.5))
        with self.assertRaisesRegex(RuntimeError, "stale"):
            preview.watchdog(feedback, now=1.7)

    def test_safety_rejects_nonfinite_and_large_jump(self):
        points = [
            TrajectoryPoint(0.0, np.zeros(6), np.zeros(3)),
            TrajectoryPoint(0.1, np.array([np.nan, 0, 0, 0, 0, 0]), np.zeros(3)),
        ]
        errors = validate_trajectory(points, np.tile([-10.0, 10.0], (6, 1)))
        self.assertTrue(any("NaN" in error for error in errors))

        points[1] = TrajectoryPoint(
            0.1, np.deg2rad([9.0, 0, 0, 0, 0, 0]), np.zeros(3)
        )
        errors = validate_trajectory(points, np.tile([-10.0, 10.0], (6, 1)))
        self.assertTrue(any("jump" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
