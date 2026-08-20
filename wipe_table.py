"""Replay the saved ``test1`` table-wiping trajectory on a CR3.

The script uses the same guarded playback path as the GUI/CLI:

1. load and validate ``trajectories/test1.json``;
2. connect to port 30004 and verify the robot is motion-ready;
3. move to the first trajectory pose through the shared low-speed ramp;
4. stream the complete trajectory as continuous ServoJ commands.

Real execution is deliberately locked until ``--enable-real-execution`` is
provided.  Unless ``--yes`` is supplied, the operator must type ``YES`` after
reviewing the trajectory summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import threading

import mujoco
import numpy as np

import run_keyboard_sim2real as core
from cr3_sim2real.joint_mapping import MAPPING_CALIBRATED
from cr3_sim2real.trajectory import TrajectoryRecorder, validate_trajectory


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAJECTORY = ROOT / "trajectories" / "test1.json"


def _percent(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= parsed < 20:
        raise argparse.ArgumentTypeError("must be between 1 and 19")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely replay trajectories/test1.json on a Dobot CR3."
    )
    parser.add_argument(
        "--robot-ip",
        default="192.168.5.11",
        help="CR3 IP address (default: 192.168.5.11)",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY,
        help="trajectory JSON (default: trajectories/test1.json)",
    )
    parser.add_argument(
        "--speed-percent",
        type=_percent,
        default=10,
        help="host SpeedJ percentage, 1..19 (default: 10)",
    )
    parser.add_argument(
        "--acceleration-percent",
        type=_percent,
        default=5,
        help="retained AccJ percentage, 1..19 (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=core.DEFAULT_MODEL,
        help="MuJoCo model used for joint-limit validation",
    )
    parser.add_argument(
        "--dobot-sdk-root",
        type=Path,
        default=ROOT,
        help="directory containing the Dobot SDK modules",
    )
    parser.add_argument(
        "--enable-real-execution",
        action="store_true",
        help="unlock the physical robot command path",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the final typed YES confirmation",
    )
    return parser


def _load_and_validate(path: Path, model_path: Path) -> TrajectoryRecorder:
    if not path.is_file():
        raise FileNotFoundError(f"trajectory not found: {path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

    recorder = TrajectoryRecorder(sample_period=1.0 / core.RECORD_RATE_HZ)
    count = recorder.load_json(path)
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    joint_ids, _qpos_indices, _dof_indices = core.joint_indices(model)
    errors = validate_trajectory(recorder.points, model.jnt_range[joint_ids])
    if errors:
        raise ValueError("trajectory safety check failed: " + "; ".join(errors))
    if not MAPPING_CALIBRATED:
        raise RuntimeError("joint mapping is not calibrated; execution is locked")
    if count == 0:  # defensive; load_json already rejects empty files
        raise ValueError("trajectory is empty")
    return recorder


def _print_summary(recorder: TrajectoryRecorder, path: Path, args) -> None:
    start_deg = core.sim_rad_to_real_deg(recorder.points[0].q)
    end_deg = core.sim_rad_to_real_deg(recorder.points[-1].q)
    max_robot_speed = core.CR3_MAX_JOINT_SPEED_DEG_S * args.speed_percent / 100.0
    print("=" * 72)
    print("WIPE TABLE TRAJECTORY READY")
    print(f"Trajectory: {path.resolve()}")
    print(f"Samples: {len(recorder.points)}   Duration: {recorder.duration:.3f} s")
    print("Start joints (deg):", np.round(start_deg, 3).tolist())
    print("End joints   (deg):", np.round(end_deg, 3).tolist())
    print(
        f"Execution: {args.robot_ip}:30003   "
        f"host speed limit: {max_robot_speed:.2f} deg/s   "
        f"SpeedJ={args.speed_percent}% AccJ={args.acceleration_percent}%"
    )
    print("The robot will first ramp to the start pose, then stream ServoJ.")
    print("=" * 72)


def main() -> int:
    args = build_parser().parse_args()
    if not args.enable_real_execution:
        print(
            "Real execution is locked. Re-run with "
            "--enable-real-execution after checking the workspace."
        )
        return 2

    trajectory_path = args.trajectory
    if not trajectory_path.is_absolute():
        trajectory_path = (ROOT / trajectory_path).resolve()
    model_path = args.model
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()

    try:
        recorder = _load_and_validate(trajectory_path, model_path)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Cannot prepare trajectory: {exc}")
        return 1

    _print_summary(recorder, trajectory_path, args)
    if not args.yes:
        try:
            answer = input("Type YES to connect and move the physical CR3: ").strip()
        except EOFError:
            answer = ""
        if answer != "YES":
            print("Cancelled; no socket was opened and no command was sent.")
            return 0

    stop_event = threading.Event()
    try:
        core.execute_real_trajectory(
            recorder,
            robot_ip=args.robot_ip,
            sdk_root=args.dobot_sdk_root.resolve(),
            speed_percent=args.speed_percent,
            acceleration_percent=args.acceleration_percent,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        stop_event.set()
        print("Interrupted; playback stop requested.")
        return 130
    except Exception as exc:
        print(f"Wipe-table playback failed: {exc}")
        return 1

    print("Wipe-table playback completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
