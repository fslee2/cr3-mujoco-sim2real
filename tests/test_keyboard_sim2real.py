import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
from mujoco.glfw import glfw
import numpy as np

import run_keyboard_sim2real as app
import run_keyboard_sim2real_gui as gui
from cr3_sim2real.hardware import (
    DEFAULT_LIVE_JOINT_SPEED_DEG_S,
    PACKET_MARKER,
    LIVE_SERVO_PERIOD_S,
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
    wait_for_drag_state,
)
from cr3_sim2real.hamer_bridge import HamerArmMapper, hamer_cam_t
from cr3_sim2real.joint_mapping import real_deg_to_sim_rad, sim_rad_to_real_deg
from cr3_sim2real.trajectory import (
    TrajectoryPoint,
    TrajectoryRecorder,
    build_jointmovj_dry_run,
    downsample_points,
    validate_trajectory,
)


class KeyboardSim2RealTests(unittest.TestCase):
    def test_gui_language_translation_switches_both_directions(self):
        controller = object.__new__(gui.CR3ControlGUI)
        controller.language = "en"
        self.assertEqual(controller._tr("使能"), "Enable")
        self.assertEqual(controller._tr("控制器状态  ", "Controller status  "), "Controller status  ")
        controller.language = "zh"
        self.assertEqual(controller._tr("使能"), "使能")
        self.assertEqual(gui.UI_TEXT_ZH["Record"], "录制后回放")

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

    def test_playback_waits_on_feedback_without_calling_sync(self):
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

            def JointMovJ(self, *args):
                self.commands.append(args)
                return "0,{},JointMovJ(...);"

        hardware = PlaybackHardware("192.0.2.1", 10, 5)
        hardware.feedback = StaticFeedback()
        hardware.dashboard = Dashboard(hardware.feedback)
        hardware.move = MoveWithoutSync()
        hardware.execute([target])
        self.assertEqual(len(hardware.move.commands), 1)
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

        with tempfile.TemporaryDirectory() as directory:
            path = recorder.save_json(Path(directory))
            self.assertTrue(path.is_file())

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

    def test_safety_rejects_nonfinite_and_large_jump(self):
        points = [
            TrajectoryPoint(0.0, np.zeros(6), np.zeros(3)),
            TrajectoryPoint(0.1, np.array([np.nan, 0, 0, 0, 0, 0]), np.zeros(3)),
        ]
        errors = validate_trajectory(points, np.tile([-10.0, 10.0], (6, 1)))
        self.assertTrue(any("NaN" in error for error in errors))

        points[1] = TrajectoryPoint(
            0.1, np.deg2rad([6.0, 0, 0, 0, 0, 0]), np.zeros(3)
        )
        errors = validate_trajectory(points, np.tile([-10.0, 10.0], (6, 1)))
        self.assertTrue(any("jump" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
