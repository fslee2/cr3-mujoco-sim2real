"""CR3 numpad Cartesian control: simulate, record, review, then dry-run.

The default mode is simulation-only. It never opens a Dobot socket and never
sends a command to the physical robot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import queue
import threading
import time

import mujoco
from mujoco.glfw import glfw
import mujoco.viewer
import numpy as np

from cr3_sim2real.joint_mapping import (
    JOINT_NAMES,
    MAPPING_CALIBRATED,
    real_deg_to_sim_rad,
    sim_rad_to_real_deg,
)
from cr3_sim2real.hardware import (
    CR3_MAX_JOINT_SPEED_DEG_S,
    DEFAULT_LIVE_JOINT_SPEED_DEG_S,
    FeedbackReceiver,
    LIVE_SERVO_PERIOD_S,
    LiveServoHardware,
    PlaybackHardware,
    STRICT_20_PERCENT_LIMIT_DEG_S,
    require_motion_ready,
)
from cr3_sim2real.trajectory import (
    TrajectoryRecorder,
    build_servoj_dry_run,
    resample_joint_trajectory,
    validate_trajectory,
)


ROOT = Path(__file__).resolve().parent
MUJOCO_ROOT = ROOT.parents[1] / "mujoco_ws"
DEFAULT_MODEL = MUJOCO_ROOT / "scenes" / "cr3_scene.xml"
DEFAULT_OUTPUT_DIR = ROOT / "trajectories"
HOME_Q_RAD = np.array([0.0, 0.6072, -1.7223, -0.2949, 1.6134, 0.0])

# Simulation responsiveness is intentionally independent from real-robot
# SpeedJ/AccJ. These values only affect the MuJoCo preview.
TRANSLATION_STEP_M = 0.010
ROTATION_STEP_RAD = np.deg2rad(2.0)
SIM_MAX_JOINT_SPEED_RAD_S = np.deg2rad(60.0)
MAX_RESOLVED_JOINT_STEP_RAD = np.deg2rad(5.0)
IK_DAMPING = 0.1
ORIENTATION_TASK_WEIGHT = 0.75
MAX_ORIENTATION_ERROR_RAD = np.deg2rad(12.0)
REAL_START_TOLERANCE_DEG = 3.0
CONTROL_RATE_HZ = 100.0
RECORD_RATE_HZ = 50.0
REAL_CONFIRMATION_WINDOW_S = 15.0


NUMPAD_TWIST = {
    glfw.KEY_KP_8: np.array([0, 0, TRANSLATION_STEP_M, 0, 0, 0]),
    glfw.KEY_KP_2: np.array([0, 0, -TRANSLATION_STEP_M, 0, 0, 0]),
    glfw.KEY_KP_4: np.array([0, -TRANSLATION_STEP_M, 0, 0, 0, 0]),
    glfw.KEY_KP_6: np.array([0, TRANSLATION_STEP_M, 0, 0, 0, 0]),
    glfw.KEY_KP_7: np.array([-TRANSLATION_STEP_M, 0, 0, 0, 0, 0]),
    glfw.KEY_KP_9: np.array([TRANSLATION_STEP_M, 0, 0, 0, 0, 0]),
    glfw.KEY_KP_1: np.array([0, 0, 0, -ROTATION_STEP_RAD, 0, 0]),
    glfw.KEY_KP_3: np.array([0, 0, 0, ROTATION_STEP_RAD, 0, 0]),
    glfw.KEY_KP_0: np.array([0, 0, 0, 0, -ROTATION_STEP_RAD, 0]),
    glfw.KEY_KP_DECIMAL: np.array([0, 0, 0, 0, ROTATION_STEP_RAD, 0]),
    glfw.KEY_KP_SUBTRACT: np.array([0, 0, 0, 0, 0, -ROTATION_STEP_RAD]),
    glfw.KEY_KP_ADD: np.array([0, 0, 0, 0, 0, ROTATION_STEP_RAD]),
}

NUMPAD_CONTROL_KEYS = {
    glfw.KEY_KP_5,
    glfw.KEY_KP_ENTER,
    glfw.KEY_KP_DIVIDE,
    glfw.KEY_KP_MULTIPLY,
}
ACCEPTED_NUMPAD_KEYS = frozenset(NUMPAD_TWIST) | NUMPAD_CONTROL_KEYS


def percent_below_twenty(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed < 20:
        raise argparse.ArgumentTypeError("must be an integer from 1 through 19")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def live_speed_below_twenty(value: str) -> float:
    parsed = positive_float(value)
    if parsed >= STRICT_20_PERCENT_LIMIT_DEG_S:
        raise argparse.ArgumentTypeError(
            "must be strictly below 36 deg/s (20% of CR3's 180 deg/s maximum)"
        )
    return parsed


def joint_indices(model: mujoco.MjModel):
    joint_ids = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {name}")
        joint_ids.append(joint_id)
    joint_ids = np.asarray(joint_ids, dtype=int)
    return (
        joint_ids,
        model.jnt_qposadr[joint_ids].astype(int),
        model.jnt_dofadr[joint_ids].astype(int),
    )


def apply_cartesian_increment(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    end_effector_id: int,
    arm_dof_indices: np.ndarray,
    arm_joint_ids: np.ndarray,
    q_target: np.ndarray,
    twist: np.ndarray,
) -> np.ndarray:
    """Existing demo's damped resolved-rate IK, factored into one operation."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, end_effector_id)
    jacobian = np.vstack(
        [jacp[:, arm_dof_indices], jacr[:, arm_dof_indices]]
    )
    dq = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + IK_DAMPING * np.eye(6), twist
    )
    dq_norm = float(np.linalg.norm(dq))
    if dq_norm > MAX_RESOLVED_JOINT_STEP_RAD:
        dq *= MAX_RESOLVED_JOINT_STEP_RAD / dq_norm

    result = q_target + dq
    return np.clip(
        result,
        model.jnt_range[arm_joint_ids, 0],
        model.jnt_range[arm_joint_ids, 1],
    )


def rotation_error_vector(
    target_rotation: np.ndarray,
    current_rotation: np.ndarray,
    *,
    max_angle_rad: float = MAX_ORIENTATION_ERROR_RAD,
) -> np.ndarray:
    """Return a bounded world-frame SO(3) error vector in radians."""
    target = np.asarray(target_rotation, dtype=float).reshape(3, 3)
    current = np.asarray(current_rotation, dtype=float).reshape(3, 3)
    relative = target @ current.T
    skew = 0.5 * (relative - relative.T)
    vee = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=float)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-7:
        error = vee
    else:
        sine = float(np.sin(angle))
        if abs(sine) > 1e-7:
            error = vee * (angle / sine)
        else:
            # The 180-degree case is rare for hand tracking; use the most
            # stable diagonal axis available rather than emitting NaNs.
            axis = np.sqrt(np.maximum(np.diag(relative) + 1.0, 0.0) * 0.5)
            axis[int(np.argmax(axis))] = max(axis[int(np.argmax(axis))], 1e-7)
            error = axis * angle
    norm = float(np.linalg.norm(error))
    if norm > max_angle_rad > 0.0:
        error *= max_angle_rad / norm
    return error


def apply_cartesian_pose_increment(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    end_effector_id: int,
    arm_dof_indices: np.ndarray,
    arm_joint_ids: np.ndarray,
    current_q: np.ndarray,
    position_delta: np.ndarray,
    target_rotation: np.ndarray,
    *,
    orientation_weight: float = ORIENTATION_TASK_WEIGHT,
) -> np.ndarray:
    """One resolved-rate XYZ+orientation step from the current simulation pose.

    This is an opt-in Quest experiment.  ``current_q`` is deliberately the
    current MuJoCo pose, not the previous target, so the orientation hold does
    not accumulate stale IK error.  The normal XYZ-only helper is unchanged.
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, end_effector_id)
    current_rotation = np.asarray(data.xmat[end_effector_id], dtype=float).reshape(3, 3)
    orientation_error = rotation_error_vector(target_rotation, current_rotation)
    weight = max(float(orientation_weight), 0.0)
    jacobian = np.vstack(
        [jacp[:, arm_dof_indices], weight * jacr[:, arm_dof_indices]]
    )
    task = np.concatenate(
        [np.asarray(position_delta, dtype=float).reshape(3), weight * orientation_error]
    )
    dq = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + IK_DAMPING * np.eye(6), task
    )
    dq_norm = float(np.linalg.norm(dq))
    if dq_norm > MAX_RESOLVED_JOINT_STEP_RAD:
        dq *= MAX_RESOLVED_JOINT_STEP_RAD / dq_norm
    result = np.asarray(current_q, dtype=float).reshape(-1) + dq
    return np.clip(
        result,
        model.jnt_range[arm_joint_ids, 0],
        model.jnt_range[arm_joint_ids, 1],
    )


def print_review(recorder: TrajectoryRecorder, saved_path: Path | None) -> None:
    points = recorder.points
    print("\n" + "=" * 64)
    print("SIMULATION TRAJECTORY READY")
    print(f"Duration: {recorder.duration:.3f} s")
    print(f"Samples: {len(points)}")
    print("Start joints (rad):", np.round(points[0].q, 5))
    print("End joints   (rad):", np.round(points[-1].q, 5))
    if saved_path is not None:
        print(f"Saved: {saved_path}")
    print("Trajectory has NOT been sent to the real robot.")
    print("Numpad * = safety check + continuous ServoJ dry run")
    print("=" * 64)


def dry_run_review(
    recorder: TrajectoryRecorder,
    joint_ranges_rad: np.ndarray,
    speed_percent: int,
    acceleration_percent: int,
) -> bool:
    errors = validate_trajectory(recorder.points, joint_ranges_rad)
    if errors:
        print("\nTRAJECTORY REJECTED")
        for error in errors:
            print(f"  - {error}")
        print("No command has been sent to the real robot.")
        return False

    commands = build_servoj_dry_run(
        recorder.points,
        sample_period=LIVE_SERVO_PERIOD_S,
    )
    print("\n" + "=" * 64)
    print("SAFETY CHECK PASSED — DRY RUN ONLY")
    print(f"Recorded samples: {len(recorder.points)}")
    print(
        f"Continuous ServoJ samples: {len(commands)} at "
        f"{1 / LIVE_SERVO_PERIOD_S:.1f} Hz"
    )
    print(
        f"Host joint speed cap={CR3_MAX_JOINT_SPEED_DEG_S * speed_percent / 100.0:.1f} deg/s  "
        f"AccJ={acceleration_percent}% reserved (ServoJ does not use AccJ)"
    )
    for index, command in enumerate(commands[:5], start=1):
        print(f"  {index:02d}: {command}")
    if len(commands) > 5:
        print(f"  ... {len(commands) - 5} more ServoJ samples")
    if not MAPPING_CALIBRATED:
        print("REAL EXECUTION LOCKED: joint sign/offset mapping is UNCALIBRATED.")
    print("No socket was opened. No command was sent to the real robot.")
    print("=" * 64)
    return True


def execute_real_trajectory(
    recorder: TrajectoryRecorder,
    *,
    robot_ip: str,
    sdk_root: Path,
    speed_percent: int,
    acceleration_percent: int,
    stop_event: threading.Event | None = None,
    feedback: FeedbackReceiver | None = None,
) -> None:
    """Execute only after all external and in-app gates have been passed."""
    if not MAPPING_CALIBRATED:
        raise RuntimeError("Real execution rejected: joint mapping is uncalibrated")
    if not 1 <= speed_percent < 20 or not 1 <= acceleration_percent < 20:
        raise RuntimeError("Real SpeedJ and AccJ must both be below 20%")
    if not sdk_root.is_dir():
        raise FileNotFoundError(f"Dobot SDK directory not found: {sdk_root}")

    hardware = PlaybackHardware(
        robot_ip,
        speed_percent=speed_percent,
        acceleration_percent=acceleration_percent,
        feedback=feedback,
    )
    try:
        state = hardware.connect()
        real_start = state.joints_deg
        expected_start = sim_rad_to_real_deg(recorder.points[0].q)
        start_error = float(np.max(np.abs(real_start - expected_start)))
        if start_error > REAL_START_TOLERANCE_DEG:
            raise RuntimeError(
                f"Real start pose differs by {start_error:.3f} deg; "
                f"limit is {REAL_START_TOLERANCE_DEG:.3f} deg"
            )
        samples = resample_joint_trajectory(
            recorder.points,
            sample_period=LIVE_SERVO_PERIOD_S,
        )
        real_stream = [
            (point.time, sim_rad_to_real_deg(point.q)) for point in samples
        ]
        hardware.execute(real_stream, stop_event=stop_event)
    finally:
        hardware.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Numpad CR3 simulation, trajectory review, and safe dry run."
    )
    parser.add_argument(
        "--mode",
        choices=("record", "live"),
        default="record",
        help="record=review then continuous ServoJ playback; live=33 Hz guarded ServoJ sync",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sim-speed-scale",
        type=positive_float,
        default=1.0,
        help="MuJoCo-only movement multiplier; does not affect real SpeedJ.",
    )
    parser.add_argument("--real-speed-percent", type=percent_below_twenty, default=10)
    parser.add_argument(
        "--real-acceleration-percent", type=percent_below_twenty, default=10
    )
    parser.add_argument(
        "--enable-real-execution",
        action="store_true",
        help="Unlock the confirmation UI; still rejected while mapping is uncalibrated.",
    )
    parser.add_argument("--robot-ip")
    parser.add_argument("--dobot-sdk-root", type=Path, default=ROOT)
    parser.add_argument(
        "--live-max-joint-speed-deg-s",
        type=live_speed_below_twenty,
        default=DEFAULT_LIVE_JOINT_SPEED_DEG_S,
    )
    args = parser.parse_args()

    if args.enable_real_execution and not args.robot_ip:
        parser.error("--enable-real-execution requires --robot-ip")
    if args.mode == "live" and not args.enable_real_execution:
        parser.error("--mode live requires --enable-real-execution")
    if args.mode == "live" and not MAPPING_CALIBRATED:
        parser.error("--mode live is locked until joint mapping is calibrated")
    if args.enable_real_execution and not MAPPING_CALIBRATED:
        parser.error("real execution is locked until joint mapping is calibrated")

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    arm_joint_ids, arm_qpos_indices, arm_dof_indices = joint_indices(model)
    end_effector_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "Link6"
    )
    if end_effector_id < 0:
        raise ValueError("MuJoCo end-effector body not found: Link6")

    live_hardware: LiveServoHardware | None = None
    if args.mode == "live":
        live_hardware = LiveServoHardware(
            args.robot_ip,
            max_joint_speed_deg_s=args.live_max_joint_speed_deg_s,
        )
        initial_feedback = live_hardware.connect()
        initial_q_rad = real_deg_to_sim_rad(initial_feedback.joints_deg)
    elif args.enable_real_execution:
        initial_receiver = FeedbackReceiver(args.robot_ip)
        try:
            initial_feedback = initial_receiver.start()
            require_motion_ready(initial_feedback)
            initial_q_rad = real_deg_to_sim_rad(initial_feedback.joints_deg)
        finally:
            initial_receiver.close()
        print("Record mode initialized from the current real CR3 pose (read-only).")
    else:
        initial_q_rad = HOME_Q_RAD

    data.qpos[arm_qpos_indices] = initial_q_rad
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    q_target = data.qpos[arm_qpos_indices].copy()
    joint_ranges_rad = model.jnt_range[arm_joint_ids].copy()

    key_events: queue.SimpleQueue[int] = queue.SimpleQueue()

    def key_callback(keycode: int) -> None:
        # Deliberately ignore every main-keyboard key.
        if keycode in ACCEPTED_NUMPAD_KEYS:
            key_events.put(keycode)

    recorder = TrajectoryRecorder(sample_period=1.0 / RECORD_RATE_HZ)
    saved_path: Path | None = None
    dry_run_passed = False
    real_confirmation_deadline = 0.0

    if args.mode == "record":
        print("MODE: RECORD_PLAYBACK")
        print("Simulation first; hardware opens only after review + confirmations.")
    else:
        print("MODE: LIVE_SYNC")
        print(
            f"ServoJ 33 Hz, host-limited to {args.live_max_joint_speed_deg_s:g} deg/s."
        )
    print(f"MuJoCo speed scale: {args.sim_speed_scale:g}x")
    print(
        f"Real command preview: SpeedJ={args.real_speed_percent}% "
        f"AccJ={args.real_acceleration_percent}%"
    )
    print("Numpad movement: 8/2=Z  4/6=Y  7/9=X")
    print("Numpad rotation: 1/3=RX  0/.=RY  -/+=RZ")
    if args.mode == "record":
        print("Numpad 5=home  Enter=start/stop record  /=clear  *=review/dry run")
    else:
        print("LIVE_SYNC: movement keys only; home/record/review keys are disabled")
    print("Close the MuJoCo window to exit.")

    last_loop_time = time.monotonic()
    try:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            viewer.cam.lookat[:] = (0.0, 0.3, 0.9)
            viewer.cam.distance = 1.6
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20

            while viewer.is_running():
                loop_started = time.monotonic()
                dt = min(max(loop_started - last_loop_time, 0.0), 0.05)
                last_loop_time = loop_started

                while not key_events.empty():
                    keycode = key_events.get()
                    if keycode in NUMPAD_TWIST:
                        q_target = apply_cartesian_increment(
                            model,
                            data,
                            end_effector_id,
                            arm_dof_indices,
                            arm_joint_ids,
                            q_target,
                            NUMPAD_TWIST[keycode] * args.sim_speed_scale,
                        )
                    elif args.mode == "live":
                        print("\nLIVE_SYNC ignores home/record/review keys")
                    elif keycode == glfw.KEY_KP_5:
                        q_target = HOME_Q_RAD.copy()
                        print("\n[SIM] Returning to home")
                    elif keycode == glfw.KEY_KP_DIVIDE:
                        recorder.clear()
                        saved_path = None
                        dry_run_passed = False
                        real_confirmation_deadline = 0.0
                        print("\nTRAJECTORY CLEARED — real robot unchanged")
                    elif keycode == glfw.KEY_KP_ENTER:
                        if recorder.recording:
                            recorder.stop(
                                loop_started,
                                data.qpos[arm_qpos_indices],
                                data.xpos[end_effector_id],
                            )
                            saved_path = recorder.save_json(args.output_dir.resolve())
                            dry_run_passed = False
                            print("\nRECORDING STOPPED")
                            print_review(recorder, saved_path)
                        elif (
                            recorder.review_ready
                            and dry_run_passed
                            and args.enable_real_execution
                        ):
                            if not MAPPING_CALIBRATED:
                                print("\nREAL EXECUTION LOCKED: mapping is uncalibrated")
                            else:
                                real_confirmation_deadline = (
                                    loop_started + REAL_CONFIRMATION_WINDOW_S
                                )
                                print(
                                    f"\nREAL EXECUTION ARMED FOR "
                                    f"{REAL_CONFIRMATION_WINDOW_S:.0f} SECONDS. "
                                    "Press Numpad * to confirm."
                                )
                        else:
                            recorder.start(
                                loop_started,
                                data.qpos[arm_qpos_indices],
                                data.xpos[end_effector_id],
                            )
                            saved_path = None
                            dry_run_passed = False
                            real_confirmation_deadline = 0.0
                            print("\nRECORDING STARTED")
                    elif keycode == glfw.KEY_KP_MULTIPLY:
                        if not recorder.review_ready:
                            print("\nNo stopped trajectory is ready for review")
                        elif real_confirmation_deadline > 0.0:
                            if real_confirmation_deadline >= loop_started:
                                print("\nFINAL NUMPAD CONFIRMATION RECEIVED")
                                real_confirmation_deadline = 0.0
                                try:
                                    execute_real_trajectory(
                                        recorder,
                                        robot_ip=args.robot_ip,
                                        sdk_root=args.dobot_sdk_root.resolve(),
                                        speed_percent=args.real_speed_percent,
                                        acceleration_percent=(
                                            args.real_acceleration_percent
                                        ),
                                    )
                                except Exception as exc:
                                    print(f"\nREAL EXECUTION ABORTED: {exc}")
                                    print(
                                        "Remaining ServoJ samples were not sent. "
                                        "MuJoCo remains open for review."
                                    )
                            else:
                                real_confirmation_deadline = 0.0
                                print(
                                    "\nREAL EXECUTION CONFIRMATION EXPIRED. "
                                    "No command was sent. Press Numpad Enter "
                                    "to arm again."
                                )
                        else:
                            dry_run_passed = dry_run_review(
                                recorder,
                                joint_ranges_rad,
                                args.real_speed_percent,
                                args.real_acceleration_percent,
                            )

                if args.mode == "live":
                    requested_real_deg = sim_rad_to_real_deg(q_target)
                    live_hardware.send_if_due(requested_real_deg, loop_started)
                    actual = live_hardware.feedback.require_fresh()
                    data.qpos[arm_qpos_indices] = real_deg_to_sim_rad(
                        actual.joints_deg
                    )
                else:
                    current_q = data.qpos[arm_qpos_indices]
                    max_step = SIM_MAX_JOINT_SPEED_RAD_S * args.sim_speed_scale * dt
                    data.qpos[arm_qpos_indices] += np.clip(
                        q_target - current_q, -max_step, max_step
                    )
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)

                recorder.sample(
                    loop_started,
                    data.qpos[arm_qpos_indices],
                    data.xpos[end_effector_id],
                )
                viewer.sync()
                time.sleep(
                    max(
                        0.0,
                        1.0 / CONTROL_RATE_HZ - (time.monotonic() - loop_started),
                    )
                )
    finally:
        if live_hardware is not None:
            live_hardware.close()

    if recorder.recording:
        recorder.stop(
            time.monotonic(),
            data.qpos[arm_qpos_indices],
            data.xpos[end_effector_id],
        )
        saved_path = recorder.save_json(args.output_dir.resolve())
        print(f"\nWindow closed; active recording saved to {saved_path}")
    print("Done. No automatic real-robot action was performed.")


if __name__ == "__main__":
    main()
