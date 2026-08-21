"""Guarded Dobot CR3 hardware access for playback and live synchronization."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import socket
import threading
import time

import numpy as np

from dobot_api import DobotApiDashboard, DobotApiMove, MyType


FEEDBACK_PORT = 30004
MOTION_PORT = 30003
DASHBOARD_PORT = 29999
PACKET_MARKER = 0x123456789ABCDEF
CR3_MAX_JOINT_SPEED_DEG_S = 180.0
STRICT_20_PERCENT_LIMIT_DEG_S = CR3_MAX_JOINT_SPEED_DEG_S * 0.20
DEFAULT_LIVE_JOINT_SPEED_DEG_S = 18.0
DEFAULT_LIVE_TRACKING_ERROR_DEG = 5.0
LIVE_SERVO_PERIOD_S = 0.03
LIVE_SERVO_T_S = 0.10
# Adaptively smooth discrete IK targets before the strict velocity limiter.
# Slow motion keeps enough smoothing to suppress steps; fast motion minimizes
# added delay so the physical arm remains responsive.
LIVE_TARGET_FILTER_SLOW_ALPHA = 0.55
LIVE_TARGET_FILTER_FAST_ALPHA = 0.90
LIVE_TARGET_FILTER_SLOW_SPEED_DEG_S = 1.0
LIVE_TARGET_FILTER_FAST_SPEED_DEG_S = 12.0
FEEDBACK_WATCHDOG_S = 0.20
FEEDBACK_CONNECT_TIMEOUT_S = 2.0
FEEDBACK_RECONNECT_DELAY_S = 0.25
PLAYBACK_POSITION_TOLERANCE_DEG = 0.20
PLAYBACK_STOP_SPEED_TOLERANCE_DEG_S = 0.50
PLAYBACK_WAYPOINT_TIMEOUT_S = 60.0
PLAYBACK_POLL_PERIOD_S = 0.02
PLAYBACK_MAX_TRACKING_ERROR_DEG = 5.0
QUEUE_START_TIMEOUT_S = 2.0
DRAG_STATE_TIMEOUT_S = 2.5


@dataclass(frozen=True)
class RobotFeedback:
    received_at: float
    joints_deg: np.ndarray
    joint_speeds_deg_s: np.ndarray
    robot_mode: int
    enabled: bool
    error: bool
    queue_running: bool = False
    paused: bool = False
    running: bool = False
    tcp_pose: np.ndarray | None = None
    tcp_speed: np.ndarray | None = None
    motor_temperatures_c: np.ndarray | None = None
    speed_scaling: float = 0.0
    velocity_ratio: int = 0
    acceleration_ratio: int = 0
    digital_input_bits: int = 0
    digital_output_bits: int = 0
    joint_currents: np.ndarray | None = None
    joint_torques: np.ndarray | None = None
    actual_tcp_force: np.ndarray | None = None
    tcp_force_from_currents: np.ndarray | None = None
    six_force_value: np.ndarray | None = None
    six_force_online: bool = False
    drag_status: bool = False
    joint_modes: np.ndarray | None = None
    payload_kg: float = 0.0
    payload_center_mm: np.ndarray | None = None


def limit_joint_velocity(
    previous_deg,
    requested_deg,
    *,
    max_joint_speed_deg_s: float,
    dt: float,
) -> np.ndarray:
    """Rate-limit one six-axis command; used before every ServoJ call."""
    if not 0.0 < max_joint_speed_deg_s < STRICT_20_PERCENT_LIMIT_DEG_S:
        raise ValueError(
            "live max joint speed must be positive and strictly below "
            f"{STRICT_20_PERCENT_LIMIT_DEG_S:.1f} deg/s"
        )
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    previous = np.asarray(previous_deg, dtype=float)
    requested = np.asarray(requested_deg, dtype=float)
    if previous.shape != (6,) or requested.shape != (6,):
        raise ValueError("ServoJ commands must contain exactly six joints")
    if not np.isfinite(previous).all() or not np.isfinite(requested).all():
        raise ValueError("ServoJ commands cannot contain NaN or Inf")
    max_delta = max_joint_speed_deg_s * dt
    return previous + np.clip(requested - previous, -max_delta, max_delta)


class FeedbackReceiver:
    """Continuously receive 30004 data, reconnecting after transport failures."""

    def __init__(self, robot_ip: str) -> None:
        self.robot_ip = robot_ip
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._latest: RobotFeedback | None = None
        self._error: Exception | None = None
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self, timeout: float = 4.0) -> RobotFeedback:
        if self._thread is not None:
            raise RuntimeError("feedback receiver already started")
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            with self._lock:
                error = self._error
            self.close()
            detail = f"; last error: {error}" if error is not None else ""
            raise TimeoutError(
                f"No CR3 feedback within {timeout:.1f} seconds{detail}"
            )
        return self.require_fresh()

    def latest(self) -> RobotFeedback | None:
        with self._lock:
            return self._latest

    def require_fresh(self, max_age: float = FEEDBACK_WATCHDOG_S) -> RobotFeedback:
        with self._lock:
            error = self._error
            feedback = self._latest
        if error is not None:
            raise RuntimeError(f"CR3 feedback failed: {error}") from error
        if feedback is None:
            raise RuntimeError("No CR3 feedback is available")
        age = time.monotonic() - feedback.received_at
        if age > max_age:
            raise RuntimeError(f"CR3 feedback is stale ({age:.3f} s)")
        return feedback

    def close(self) -> None:
        self._stop.set()
        sock = self._socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def _receive_loop(self) -> None:
        packet_size = MyType.itemsize
        marker_offset = MyType.fields["test_value"][1]
        marker_bytes = PACKET_MARKER.to_bytes(8, byteorder="little")
        while not self._stop.is_set():
            buffer = bytearray()
            try:
                with socket.create_connection(
                    (self.robot_ip, FEEDBACK_PORT),
                    timeout=FEEDBACK_CONNECT_TIMEOUT_S,
                ) as sock:
                    self._socket = sock
                    sock.settimeout(0.5)
                    while not self._stop.is_set():
                        try:
                            chunk = sock.recv(packet_size * 10)
                        except socket.timeout:
                            continue
                        if not chunk:
                            raise ConnectionError("CR3 closed port 30004")
                        buffer.extend(chunk)
                        while True:
                            marker_index = buffer.find(marker_bytes)
                            if marker_index < 0:
                                if len(buffer) > packet_size * 2:
                                    del buffer[:-packet_size]
                                break
                            packet_start = marker_index - marker_offset
                            if packet_start < 0:
                                del buffer[: marker_index + len(marker_bytes)]
                                continue
                            packet_end = packet_start + packet_size
                            if len(buffer) < packet_end:
                                if packet_start > 0:
                                    del buffer[:packet_start]
                                break
                            raw = bytes(buffer[packet_start:packet_end])
                            del buffer[:packet_end]
                            parsed = np.frombuffer(raw, dtype=MyType, count=1)
                            feedback = RobotFeedback(
                                received_at=time.monotonic(),
                                joints_deg=parsed["q_actual"][0].copy(),
                                joint_speeds_deg_s=parsed["qd_actual"][0].copy(),
                                robot_mode=int(parsed["robot_mode"][0]),
                                enabled=bool(parsed["enable_status"][0]),
                                error=bool(parsed["error_status"][0]),
                                queue_running=bool(parsed["run_queued_cmd"][0]),
                                paused=bool(parsed["pause_cmd_flag"][0]),
                                running=bool(parsed["running_status"][0]),
                                tcp_pose=parsed["tool_vector_actual"][0].copy(),
                                tcp_speed=parsed["TCP_speed_actual"][0].copy(),
                                motor_temperatures_c=parsed["motor_temperatures"][0].copy(),
                                speed_scaling=float(parsed["speed_scaling"][0]),
                                velocity_ratio=int(parsed["velocity_ratio"][0]),
                                acceleration_ratio=int(parsed["acceleration_ratio"][0]),
                                digital_input_bits=int(parsed["digital_input_bits"][0]),
                                digital_output_bits=int(parsed["digital_output_bits"][0]),
                                joint_currents=parsed["i_actual"][0].copy(),
                                joint_torques=parsed["m_actual"][0].copy(),
                                actual_tcp_force=parsed["actual_TCP_force"][0].copy(),
                                tcp_force_from_currents=parsed["TCP_force"][0].copy(),
                                six_force_value=parsed["six_force_value"][0].copy(),
                                six_force_online=bool(parsed["six_force_online"][0]),
                                drag_status=bool(parsed["drag_status"][0]),
                                joint_modes=parsed["joint_modes"][0].copy(),
                                payload_kg=float(parsed["load"][0]),
                                payload_center_mm=np.asarray(
                                    [
                                        parsed["center_x"][0],
                                        parsed["center_y"][0],
                                        parsed["center_z"][0],
                                    ],
                                    dtype=float,
                                ),
                            )
                            with self._lock:
                                self._latest = feedback
                                self._error = None
                            self._ready.set()
            except Exception as exc:
                if self._stop.is_set():
                    break
                with self._lock:
                    self._error = exc
            finally:
                self._socket = None
            self._stop.wait(FEEDBACK_RECONNECT_DELAY_S)


def require_motion_ready(feedback: RobotFeedback) -> None:
    if feedback.error or feedback.robot_mode == 9:
        raise RuntimeError("CR3 has an active error/alarm")
    if not feedback.enabled:
        raise RuntimeError("CR3 is not enabled; enable it manually in DobotStudio")
    if feedback.robot_mode not in (5, 7):
        raise RuntimeError(
            f"CR3 robot mode {feedback.robot_mode} is not enabled-idle/running"
        )


def normalize_reply(reply: str | bytes) -> str:
    if isinstance(reply, bytes):
        return reply.decode("utf-8", errors="replace")
    return reply or ""


def require_command_success(command: str, reply: str | bytes) -> None:
    """Raise immediately when a Dobot command returns a non-zero ErrorID."""
    reply_text = normalize_reply(reply)
    match = re.match(r"\s*(-?\d+)\s*,", reply_text)
    if match is None:
        raise RuntimeError(f"{command} returned an unrecognized reply: {reply!r}")
    error_id = int(match.group(1))
    if error_id != 0:
        raise RuntimeError(
            f"{command} rejected by CR3 (ErrorID {error_id}): {reply_text.strip()}"
        )


def _run_dashboard_command(robot_ip: str, command: str) -> str:
    """Run one allow-listed dashboard command on a short-lived socket."""
    dashboard = DobotApiDashboard(robot_ip, DASHBOARD_PORT)
    try:
        method = getattr(dashboard, command)
        reply = method()
        require_command_success(command, reply)
        return normalize_reply(reply)
    finally:
        dashboard.close()


def clear_robot_error(robot_ip: str) -> str:
    """Clear controller alarms without automatically resuming the queue."""
    return _run_dashboard_command(robot_ip, "ClearError")


def pause_robot_queue(robot_ip: str) -> str:
    """Pause the controller command queue."""
    return _run_dashboard_command(robot_ip, "pause")


def continue_robot_queue(robot_ip: str) -> str:
    """Resume the controller command queue after explicit user confirmation."""
    return _run_dashboard_command(robot_ip, "Continue")


def set_robot_speed_factor(robot_ip: str, percent: int) -> str:
    """Set the CR3 global speed factor with this project's <20% safety cap."""
    if isinstance(percent, bool) or not isinstance(percent, (int, np.integer)):
        raise ValueError("Global speed factor must be an integer percent")
    if not 1 <= int(percent) < 20:
        raise ValueError("Global speed factor must be in [1, 20)")
    dashboard = DobotApiDashboard(robot_ip, DASHBOARD_PORT)
    try:
        reply = dashboard.SpeedFactor(int(percent))
        require_command_success("SpeedFactor", reply)
        return normalize_reply(reply)
    finally:
        dashboard.close()


def start_drag_mode(robot_ip: str) -> str:
    """Enter CR drag/teaching mode after an explicit GUI action."""
    return _run_dashboard_command(robot_ip, "StartDrag")


def stop_drag_mode(robot_ip: str) -> str:
    """Exit CR drag/teaching mode."""
    return _run_dashboard_command(robot_ip, "StopDrag")


def wait_for_drag_state(
    feedback: FeedbackReceiver,
    expected: bool,
    *,
    timeout: float = DRAG_STATE_TIMEOUT_S,
) -> RobotFeedback:
    """Wait until 30004 verifies entry to or exit from drag mode."""
    deadline = time.monotonic() + timeout
    latest: RobotFeedback | None = None
    while time.monotonic() < deadline:
        latest = feedback.require_fresh(max_age=0.5)
        dragging = latest.drag_status or latest.robot_mode in (6, 8)
        if dragging is expected:
            return latest
        time.sleep(PLAYBACK_POLL_PERIOD_S)
    mode = "enter" if expected else "exit"
    actual = "unknown"
    if latest is not None:
        actual = f"Mode={latest.robot_mode}, Drag={int(latest.drag_status)}"
    raise TimeoutError(
        f"CR3 did not {mode} drag mode within {timeout:.1f} s ({actual})"
    )


def get_robot_error_ids(robot_ip: str) -> tuple[list[list[int]], str]:
    """Read GetErrorID and return its controller/servo alarm groups."""
    reply = _run_dashboard_command(robot_ip, "GetErrorID")
    start = reply.find("{")
    end = reply.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"GetErrorID returned an invalid payload: {reply!r}")
    try:
        payload = json.loads(reply[start + 1 : end])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GetErrorID returned invalid JSON: {reply!r}") from exc
    if not isinstance(payload, list) or any(not isinstance(group, list) for group in payload):
        raise RuntimeError(f"GetErrorID returned an unexpected payload: {payload!r}")
    try:
        groups = [[int(error_id) for error_id in group] for group in payload]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"GetErrorID returned non-integer alarm IDs: {payload!r}") from exc
    return groups, reply


def enable_robot(robot_ip: str) -> str:
    """Send an explicit EnableRobot command and close the dashboard socket."""
    dashboard = DobotApiDashboard(robot_ip, DASHBOARD_PORT)
    try:
        reply = dashboard.EnableRobot()
        require_command_success("EnableRobot", reply)
        return normalize_reply(reply)
    finally:
        dashboard.close()


def enable_robot_with_payload(
    robot_ip: str,
    load_kg: float,
    center_x_mm: float,
    center_y_mm: float,
    center_z_mm: float,
) -> str:
    """Enable CR3 while applying its payload and center-of-mass parameters.

    The CR-series V3 dashboard API accepts the payload as optional arguments
    to ``EnableRobot``.  Keeping this as a separate guarded operation avoids
    silently changing the normal no-argument enable path.
    """
    values = np.asarray(
        [load_kg, center_x_mm, center_y_mm, center_z_mm], dtype=float
    )
    if not np.isfinite(values).all():
        raise ValueError("payload values must be finite")
    if not 0.0 <= float(load_kg) <= 3.0:
        raise ValueError("CR3 payload must be between 0 and 3 kg")
    if np.any(np.abs(values[1:]) > 999.0):
        raise ValueError("payload center coordinates must be within ±999 mm")
    dashboard = DobotApiDashboard(robot_ip, DASHBOARD_PORT)
    try:
        reply = dashboard.EnableRobot(
            float(load_kg),
            float(center_x_mm),
            float(center_y_mm),
            float(center_z_mm),
        )
        require_command_success("EnableRobot(payload)", reply)
        return normalize_reply(reply)
    finally:
        dashboard.close()


def disable_robot(
    robot_ip: str,
    *,
    feedback: FeedbackReceiver | None = None,
    verification_timeout: float = 3.0,
) -> str:
    """Disable the robot, verifying 30004 if 29999 closes without a reply."""
    dashboard = DobotApiDashboard(robot_ip, DASHBOARD_PORT)
    try:
        reply = dashboard.DisableRobot()
    finally:
        dashboard.close()

    try:
        require_command_success("DisableRobot", reply)
        return normalize_reply(reply)
    except RuntimeError:
        if reply not in (b"", ""):
            raise

    owned_feedback = feedback is None
    receiver = feedback if feedback is not None else FeedbackReceiver(robot_ip)
    try:
        if owned_feedback:
            receiver.start()
        deadline = time.monotonic() + verification_timeout
        while time.monotonic() < deadline:
            state = receiver.require_fresh(max_age=0.5)
            if not state.enabled:
                return "0,{verified enable_status=0 on 30004},DisableRobot();"
            time.sleep(PLAYBACK_POLL_PERIOD_S)
    finally:
        if owned_feedback:
            receiver.close()
    raise RuntimeError(
        "DisableRobot returned an empty 29999 reply and 30004 still reports "
        "enable_status=1"
    )


def power_on_robot(robot_ip: str) -> str:
    """Send an explicit PowerOn command and close the dashboard socket."""
    dashboard = DobotApiDashboard(robot_ip, DASHBOARD_PORT)
    try:
        reply = dashboard.PowerOn()
        require_command_success("PowerOn", reply)
        return normalize_reply(reply)
    finally:
        dashboard.close()


def emergency_stop_robot(robot_ip: str) -> str:
    """Trigger the controller's software EmergencyStop command."""
    dashboard = DobotApiDashboard(robot_ip, DASHBOARD_PORT)
    try:
        reply = dashboard.EmergencyStop()
        require_command_success("EmergencyStop", reply)
        return normalize_reply(reply)
    finally:
        dashboard.close()


def ensure_command_queue_running(
    feedback: FeedbackReceiver,
    dashboard: DobotApiDashboard,
    *,
    timeout: float = QUEUE_START_TIMEOUT_S,
) -> RobotFeedback:
    """Start a stopped command queue after an explicit real-motion action."""
    state = feedback.require_fresh()
    require_motion_ready(state)
    if state.queue_running:
        return state

    print("CR3 command queue is stopped; sending Continue() after confirmation")
    continue_reply = dashboard.Continue()
    require_command_success("Continue", continue_reply)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = feedback.require_fresh()
        require_motion_ready(state)
        if state.queue_running:
            print("CR3 command queue is running")
            return state
        time.sleep(PLAYBACK_POLL_PERIOD_S)
    raise TimeoutError(
        "Continue() was accepted, but CR3 run_queued_cmd remained 0; "
        "no motion command was sent"
    )


def joint_target_reached(
    feedback: RobotFeedback,
    target_deg,
    *,
    position_tolerance_deg: float = PLAYBACK_POSITION_TOLERANCE_DEG,
    stop_speed_tolerance_deg_s: float = PLAYBACK_STOP_SPEED_TOLERANCE_DEG_S,
) -> bool:
    """Return whether fresh feedback is stopped at a six-axis target."""
    target = np.asarray(target_deg, dtype=float)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("Playback target must contain six finite joints")
    if position_tolerance_deg <= 0.0 or stop_speed_tolerance_deg_s <= 0.0:
        raise ValueError("Playback arrival tolerances must be positive")
    return bool(
        np.max(np.abs(feedback.joints_deg - target))
        <= position_tolerance_deg
        and np.max(np.abs(feedback.joint_speeds_deg_s))
        <= stop_speed_tolerance_deg_s
    )


class PlaybackHardware:
    """Execute a timestamped trajectory as one continuous low-speed ServoJ stream."""

    def __init__(
        self,
        robot_ip: str,
        speed_percent: int,
        acceleration_percent: int,
        *,
        feedback: FeedbackReceiver | None = None,
    ):
        if not 1 <= speed_percent < 20 or not 1 <= acceleration_percent < 20:
            raise ValueError("Playback SpeedJ and AccJ must both be in [1, 20)")
        self.robot_ip = robot_ip
        self.speed_percent = speed_percent
        self.acceleration_percent = acceleration_percent
        self.max_joint_speed_deg_s = (
            CR3_MAX_JOINT_SPEED_DEG_S * speed_percent / 100.0
        )
        self.feedback = (
            feedback if feedback is not None else FeedbackReceiver(robot_ip)
        )
        self._owns_feedback = feedback is None
        self.dashboard: DobotApiDashboard | None = None
        self.move: DobotApiMove | None = None
        self._last_command_deg: np.ndarray | None = None

    def connect(self) -> RobotFeedback:
        try:
            state = (
                self.feedback.start()
                if self._owns_feedback
                else self.feedback.require_fresh(max_age=0.5)
            )
            require_motion_ready(state)
            self._last_command_deg = state.joints_deg.copy()
            self.dashboard = DobotApiDashboard(self.robot_ip, DASHBOARD_PORT)
            self.move = DobotApiMove(self.robot_ip, MOTION_PORT)
            return state
        except Exception:
            self.close()
            raise

    def execute(
        self,
        timed_waypoints_deg: list[tuple[float, np.ndarray]],
        *,
        stop_event: threading.Event | None = None,
    ) -> None:
        if (
            self.dashboard is None
            or self.move is None
            or self._last_command_deg is None
        ):
            raise RuntimeError("Playback hardware is not connected")
        stream = self._validate_stream(timed_waypoints_deg)
        if not stream:
            raise ValueError("Playback stream is empty")
        self.ensure_queue_running()

        started_at = time.monotonic()
        previous_stream_time = 0.0
        for index, (stream_time, requested) in enumerate(stream):
            elapsed = time.monotonic() - started_at
            if index < len(stream) - 1 and stream_time < elapsed - LIVE_SERVO_PERIOD_S:
                continue
            self._wait_until(started_at + stream_time, stop_event)
            feedback = self.feedback.require_fresh()
            require_motion_ready(feedback)
            self._require_tracking_within_limit(feedback)
            control_dt = (
                LIVE_SERVO_PERIOD_S
                if stream_time == 0.0
                else stream_time - previous_stream_time
            )
            self._send_servo_target(requested, control_dt)
            previous_stream_time = stream_time

        final_target = stream[-1][1]
        next_send_at = time.monotonic() + LIVE_SERVO_PERIOD_S
        while np.max(np.abs(final_target - self._last_command_deg)) > 1e-6:
            self._wait_until(next_send_at, stop_event)
            feedback = self.feedback.require_fresh()
            require_motion_ready(feedback)
            self._require_tracking_within_limit(feedback)
            self._send_servo_target(final_target, LIVE_SERVO_PERIOD_S)
            next_send_at = max(
                next_send_at + LIVE_SERVO_PERIOD_S,
                time.monotonic() + LIVE_SERVO_PERIOD_S,
            )

        arrived = self.wait_for_joint_arrival(final_target, stop_event=stop_event)
        max_error = float(np.max(np.abs(arrived.joints_deg - final_target)))
        print(
            f"Continuous ServoJ playback complete: {len(stream)} samples "
            f"(max final joint error {max_error:.3f} deg)"
        )

    @staticmethod
    def _validate_stream(
        timed_waypoints_deg: list[tuple[float, np.ndarray]],
    ) -> list[tuple[float, np.ndarray]]:
        stream: list[tuple[float, np.ndarray]] = []
        previous_time = float("-inf")
        for timestamp, waypoint in timed_waypoints_deg:
            timestamp = float(timestamp)
            joints = np.asarray(waypoint, dtype=float)
            if (
                not np.isfinite(timestamp)
                or timestamp < 0.0
                or timestamp <= previous_time
            ):
                raise ValueError(
                    "Playback timestamps must be finite and strictly increasing"
                )
            if joints.shape != (6,) or not np.isfinite(joints).all():
                raise ValueError("Invalid playback waypoint")
            stream.append((timestamp, joints.copy()))
            previous_time = timestamp
        return stream

    def _send_servo_target(self, requested_deg: np.ndarray, control_dt: float) -> None:
        assert self.move is not None and self._last_command_deg is not None
        planned = limit_joint_velocity(
            self._last_command_deg,
            requested_deg,
            max_joint_speed_deg_s=self.max_joint_speed_deg_s,
            dt=control_dt,
        )
        reply = self.move.ServoJ(
            *planned,
            t=LIVE_SERVO_T_S,
            lookahead_time=50,
            gain=500,
        )
        require_command_success("ServoJ", reply)
        self._last_command_deg = planned

    def _require_tracking_within_limit(self, feedback: RobotFeedback) -> None:
        assert self._last_command_deg is not None
        tracking_error = float(
            np.max(np.abs(feedback.joints_deg - self._last_command_deg))
        )
        if tracking_error > PLAYBACK_MAX_TRACKING_ERROR_DEG:
            raise RuntimeError(
                f"CR3 playback tracking error {tracking_error:.3f} deg exceeds "
                f"{PLAYBACK_MAX_TRACKING_ERROR_DEG:.3f} deg"
            )

    @staticmethod
    def _wait_until(deadline: float, stop_event: threading.Event | None) -> None:
        while True:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Playback cancelled by local emergency stop")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(remaining, PLAYBACK_POLL_PERIOD_S))

    def ensure_queue_running(
        self, timeout: float = QUEUE_START_TIMEOUT_S
    ) -> RobotFeedback:
        """Start a paused command queue after the user's final confirmation."""
        if self.dashboard is None:
            raise RuntimeError("Playback dashboard is not connected")
        return ensure_command_queue_running(
            self.feedback, self.dashboard, timeout=timeout
        )

    def wait_for_joint_arrival(
        self,
        target_deg,
        timeout: float = PLAYBACK_WAYPOINT_TIMEOUT_S,
        stop_event: threading.Event | None = None,
    ) -> RobotFeedback:
        """Wait for QActual to settle at target without relying on Sync()."""
        target = np.asarray(target_deg, dtype=float)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Invalid playback waypoint")
        deadline = time.monotonic() + timeout
        latest: RobotFeedback | None = None
        while time.monotonic() < deadline:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Playback cancelled by local emergency stop")
            latest = self.feedback.require_fresh()
            require_motion_ready(latest)
            self._require_tracking_within_limit(latest)
            if joint_target_reached(latest, target):
                return latest
            time.sleep(PLAYBACK_POLL_PERIOD_S)

        if latest is None:
            raise TimeoutError("No CR3 feedback while waiting for playback waypoint")
        max_error = float(np.max(np.abs(latest.joints_deg - target)))
        max_speed = float(np.max(np.abs(latest.joint_speeds_deg_s)))
        raise TimeoutError(
            f"CR3 did not reach playback waypoint within {timeout:.1f} s "
            f"(max error {max_error:.3f} deg, max speed {max_speed:.3f} deg/s)"
        )

    def close(self) -> None:
        if self.move is not None:
            self.move.close()
            self.move = None
        if self.dashboard is not None:
            self.dashboard.close()
            self.dashboard = None
        if self._owns_feedback:
            self.feedback.close()


class LiveServoHardware:
    """Fixed-rate ServoJ stream with feedback watchdog and speed limiting.

    The controller recommends a 30 ms secondary-development cycle. The worker
    stays independent from GUI rendering so camera and Tk work cannot insert
    gaps into the robot command stream.
    """

    def __init__(
        self,
        robot_ip: str,
        *,
        max_joint_speed_deg_s: float = DEFAULT_LIVE_JOINT_SPEED_DEG_S,
        max_tracking_error_deg: float = DEFAULT_LIVE_TRACKING_ERROR_DEG,
        feedback: FeedbackReceiver | None = None,
    ) -> None:
        if not 0.0 < max_joint_speed_deg_s < STRICT_20_PERCENT_LIMIT_DEG_S:
            raise ValueError("Live speed must be positive and strictly below 36 deg/s")
        self.robot_ip = robot_ip
        self.max_joint_speed_deg_s = max_joint_speed_deg_s
        if not np.isfinite(max_tracking_error_deg) or max_tracking_error_deg <= 0.0:
            raise ValueError("Live tracking error limit must be positive and finite")
        self.max_tracking_error_deg = float(max_tracking_error_deg)
        self.feedback = feedback if feedback is not None else FeedbackReceiver(robot_ip)
        self._owns_feedback = feedback is None
        self.dashboard: DobotApiDashboard | None = None
        self.move: DobotApiMove | None = None
        self._last_command_deg: np.ndarray | None = None
        self._filtered_target_deg: np.ndarray | None = None
        self._previous_requested_deg: np.ndarray | None = None
        self._last_send_time = 0.0
        self.last_planned_speed_deg_s = 0.0
        self.last_target_filter_alpha = LIVE_TARGET_FILTER_SLOW_ALPHA
        self._stream_lock = threading.Lock()
        self._stream_target_deg: np.ndarray | None = None
        self._stream_error: Exception | None = None
        self._stream_stop = threading.Event()
        self._stream_thread: threading.Thread | None = None

    def set_max_joint_speed(self, max_joint_speed_deg_s: float) -> None:
        """Update the host-side live speed limit while synchronization runs."""
        if not 0.0 < max_joint_speed_deg_s < STRICT_20_PERCENT_LIMIT_DEG_S:
            raise ValueError("Live speed must be positive and strictly below 36 deg/s")
        self.max_joint_speed_deg_s = float(max_joint_speed_deg_s)

    def connect(self) -> RobotFeedback:
        try:
            state = (
                self.feedback.start()
                if self._owns_feedback
                else self.feedback.require_fresh(max_age=0.5)
            )
            require_motion_ready(state)
            self._last_command_deg = state.joints_deg.copy()
            self._filtered_target_deg = state.joints_deg.copy()
            self._previous_requested_deg = state.joints_deg.copy()
            self._last_send_time = time.monotonic()
            self.dashboard = DobotApiDashboard(self.robot_ip, DASHBOARD_PORT)
            self.move = DobotApiMove(self.robot_ip, MOTION_PORT)
            ensure_command_queue_running(self.feedback, self.dashboard)
            return state
        except Exception:
            self.close()
            raise

    def send_if_due(self, requested_deg, now: float | None = None) -> np.ndarray:
        if self.move is None or self._last_command_deg is None:
            raise RuntimeError("Live hardware is not connected")
        now = time.monotonic() if now is None else now
        elapsed = now - self._last_send_time
        if elapsed < LIVE_SERVO_PERIOD_S:
            return self._last_command_deg.copy()

        feedback = self.feedback.require_fresh()
        require_motion_ready(feedback)
        tracking_error = float(
            np.max(np.abs(feedback.joints_deg - self._last_command_deg))
        )
        if tracking_error > self.max_tracking_error_deg:
            raise RuntimeError(
                f"CR3 live tracking error {tracking_error:.3f} deg exceeds "
                f"{self.max_tracking_error_deg:.3f} deg"
            )
        requested = np.asarray(requested_deg, dtype=float)
        if requested.shape != (6,) or not np.isfinite(requested).all():
            raise ValueError("Invalid live ServoJ target")
        if self._filtered_target_deg is None:
            self._filtered_target_deg = requested.copy()
            filter_alpha = LIVE_TARGET_FILTER_FAST_ALPHA
        else:
            if self._previous_requested_deg is None:
                input_speed_deg_s = LIVE_TARGET_FILTER_FAST_SPEED_DEG_S
            else:
                input_speed_deg_s = float(
                    np.max(np.abs(requested - self._previous_requested_deg))
                    / LIVE_SERVO_PERIOD_S
                )
            speed_blend = float(
                np.clip(
                    (
                        input_speed_deg_s
                        - LIVE_TARGET_FILTER_SLOW_SPEED_DEG_S
                    )
                    / (
                        LIVE_TARGET_FILTER_FAST_SPEED_DEG_S
                        - LIVE_TARGET_FILTER_SLOW_SPEED_DEG_S
                    ),
                    0.0,
                    1.0,
                )
            )
            filter_alpha = (
                LIVE_TARGET_FILTER_SLOW_ALPHA
                + speed_blend
                * (
                    LIVE_TARGET_FILTER_FAST_ALPHA
                    - LIVE_TARGET_FILTER_SLOW_ALPHA
                )
            )
            self._filtered_target_deg += filter_alpha * (
                requested - self._filtered_target_deg
            )
        self._previous_requested_deg = requested.copy()
        self.last_target_filter_alpha = float(filter_alpha)
        filtered_target = self._filtered_target_deg.copy()
        if np.max(np.abs(filtered_target - self._last_command_deg)) < 1e-6:
            # Idle wall time must not enlarge the first step of the next move.
            self._last_send_time = now
            self.last_planned_speed_deg_s = 0.0
            return self._last_command_deg.copy()
        # Always use one nominal command period. Using elapsed wall time here
        # would let an idle pause or slow render frame bypass the speed limit.
        control_dt = LIVE_SERVO_PERIOD_S
        planned = limit_joint_velocity(
            self._last_command_deg,
            filtered_target,
            max_joint_speed_deg_s=self.max_joint_speed_deg_s,
            dt=control_dt,
        )
        servo_reply = self.move.ServoJ(
            *planned, t=LIVE_SERVO_T_S, lookahead_time=50, gain=500
        )
        require_command_success("ServoJ", servo_reply)
        self.last_planned_speed_deg_s = float(
            np.max(np.abs(planned - self._last_command_deg)) / control_dt
        )
        self._last_command_deg = planned
        self._last_send_time = now
        return planned.copy()

    def latest_command_deg(self) -> np.ndarray:
        """Return the latest rate-limited target actually accepted for ServoJ."""
        if self._last_command_deg is None:
            raise RuntimeError("Live hardware is not connected")
        return self._last_command_deg.copy()

    def start_stream(self, requested_deg) -> None:
        """Start a fixed 33 Hz latest-target ServoJ worker."""
        if self.move is None or self._last_command_deg is None:
            raise RuntimeError("Live hardware is not connected")
        self.set_stream_target(requested_deg)
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return
        with self._stream_lock:
            self._stream_error = None
        self._stream_stop.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="cr3-servoj-stream",
            daemon=True,
        )
        self._stream_thread.start()

    def set_stream_target(self, requested_deg) -> None:
        """Publish the newest six-joint target without blocking the caller."""
        requested = np.asarray(requested_deg, dtype=float)
        if requested.shape != (6,) or not np.isfinite(requested).all():
            raise ValueError("Invalid live ServoJ target")
        with self._stream_lock:
            self._stream_target_deg = requested.copy()

    def raise_stream_error(self) -> None:
        """Propagate a worker failure on the GUI thread."""
        with self._stream_lock:
            error = self._stream_error
        if error is not None:
            raise RuntimeError(f"CR3 ServoJ stream stopped: {error}") from error

    def _stream_loop(self) -> None:
        next_send_at = time.monotonic() + LIVE_SERVO_PERIOD_S
        while not self._stream_stop.is_set():
            wait_s = max(0.0, next_send_at - time.monotonic())
            if self._stream_stop.wait(wait_s):
                break
            with self._stream_lock:
                target = (
                    None
                    if self._stream_target_deg is None
                    else self._stream_target_deg.copy()
                )
            if target is not None:
                try:
                    self.send_if_due(target, now=time.monotonic())
                except Exception as exc:
                    with self._stream_lock:
                        self._stream_error = exc
                    self._stream_stop.set()
                    break
            next_send_at += LIVE_SERVO_PERIOD_S
            # Do not burst stale commands after any network delay.
            now = time.monotonic()
            if next_send_at < now:
                next_send_at = now + LIVE_SERVO_PERIOD_S

    def close(self) -> None:
        self._stream_stop.set()
        stream_thread = self._stream_thread
        # Close 30003 first to release a worker blocked waiting for a reply.
        if self.move is not None:
            self.move.close()
            self.move = None
        if (
            stream_thread is not None
            and stream_thread is not threading.current_thread()
        ):
            stream_thread.join(timeout=1.0)
        self._stream_thread = None
        if self.dashboard is not None:
            self.dashboard.close()
            self.dashboard = None
        if self._owns_feedback:
            self.feedback.close()


def stream_live_target_until_reached(
    hardware: LiveServoHardware,
    target_deg,
    *,
    stop_event: threading.Event | None = None,
    timeout: float = 90.0,
) -> RobotFeedback:
    """Move to one joint target through the guarded live ServoJ limiter.

    This is used for explicit low-speed preparation moves such as returning
    both HaMeR control sides to the shared Home pose.  It never bypasses
    ``LiveServoHardware.send_if_due`` or the 30004 feedback watchdog.
    """
    target = np.asarray(target_deg, dtype=float)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("Live target must contain six finite joints")
    if not np.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("Live target timeout must be positive and finite")

    deadline = time.monotonic() + float(timeout)
    latest: RobotFeedback | None = None
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Live target move cancelled")
        latest = hardware.feedback.require_fresh()
        require_motion_ready(latest)
        if joint_target_reached(latest, target):
            return latest
        hardware.send_if_due(target)
        time.sleep(0.005)

    if latest is None:
        raise TimeoutError("No CR3 feedback while moving to live target")
    error = float(np.max(np.abs(latest.joints_deg - target)))
    raise TimeoutError(
        f"CR3 did not reach live target within {timeout:.1f} s "
        f"(max error {error:.3f} deg)"
    )
