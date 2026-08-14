"""Trajectory recording, review, conversion, and safety checks for CR3."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from cr3_sim2real.joint_mapping import JOINT_NAMES, mapping_metadata, sim_rad_to_real_deg


@dataclass(frozen=True)
class TrajectoryPoint:
    time: float
    q: np.ndarray
    tcp: np.ndarray


class TrajectoryRecorder:
    def __init__(self, sample_period: float = 0.02) -> None:
        self.sample_period = sample_period
        self.points: list[TrajectoryPoint] = []
        self.recording = False
        self.review_ready = False
        self._started_at = 0.0
        self._last_sample_time = float("-inf")

    def start(self, now: float, q, tcp) -> None:
        self.points.clear()
        self.recording = True
        self.review_ready = False
        self._started_at = now
        self._last_sample_time = float("-inf")
        self.sample(now, q, tcp, force=True)

    def sample(self, now: float, q, tcp, *, force: bool = False) -> bool:
        if not self.recording:
            return False
        relative_time = now - self._started_at
        if not force and relative_time - self._last_sample_time < self.sample_period:
            return False
        point = TrajectoryPoint(
            time=float(relative_time),
            q=np.asarray(q, dtype=float).copy(),
            tcp=np.asarray(tcp, dtype=float).copy(),
        )
        if force and self.points and relative_time <= self._last_sample_time:
            self.points[-1] = point
        else:
            self.points.append(point)
        self._last_sample_time = relative_time
        return True

    def stop(self, now: float, q, tcp) -> None:
        if not self.recording:
            return
        self.sample(now, q, tcp, force=True)
        self.recording = False
        self.review_ready = bool(self.points)

    def clear(self) -> None:
        self.points.clear()
        self.recording = False
        self.review_ready = False

    @property
    def duration(self) -> float:
        return self.points[-1].time if self.points else 0.0

    def save_json(self, output_dir: Path) -> Path:
        if not self.points:
            raise ValueError("Cannot save an empty trajectory")
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"cr3_trajectory_{stamp}.json"
        payload = {
            "format": "cr3_mujoco_joint_trajectory_v1",
            "joint_names": list(JOINT_NAMES),
            "units": {"time": "s", "q": "rad", "tcp": "m"},
            "mapping": mapping_metadata(),
            "points": [
                {
                    "time": point.time,
                    "q": point.q.tolist(),
                    "tcp": point.tcp.tolist(),
                }
                for point in self.points
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


def validate_trajectory(
    points: list[TrajectoryPoint],
    joint_ranges_rad: np.ndarray,
    *,
    max_joint_jump_deg: float = 5.0,
) -> list[str]:
    """Return rejection reasons; an empty list means the trajectory passed."""
    errors: list[str] = []
    if not points:
        return ["trajectory is empty"]

    q = np.asarray([point.q for point in points], dtype=float)
    times = np.asarray([point.time for point in points], dtype=float)
    ranges = np.asarray(joint_ranges_rad, dtype=float)

    if q.ndim != 2 or q.shape[1:] != (6,):
        errors.append(f"joint data must have shape (N, 6), got {q.shape}")
        return errors
    if not np.isfinite(q).all() or not np.isfinite(times).all():
        errors.append("trajectory contains NaN or Inf")
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        errors.append("timestamps must increase strictly")
    if ranges.shape != (6, 2):
        errors.append(f"joint ranges must have shape (6, 2), got {ranges.shape}")
    elif np.any(q < ranges[:, 0] - 1e-9) or np.any(q > ranges[:, 1] + 1e-9):
        errors.append("trajectory exceeds MuJoCo joint limits")
    if len(q) > 1:
        largest_jump = float(np.rad2deg(np.max(np.abs(np.diff(q, axis=0)))))
        if largest_jump > max_joint_jump_deg:
            errors.append(
                f"adjacent joint jump {largest_jump:.3f} deg exceeds "
                f"{max_joint_jump_deg:.3f} deg"
            )
    return errors


def downsample_points(
    points: list[TrajectoryPoint],
    minimum_period: float = 0.25,
    minimum_joint_change_deg: float = 0.05,
) -> list[TrajectoryPoint]:
    if not points:
        return []
    selected = [points[0]]
    for point in points[1:-1]:
        joint_change_deg = float(
            np.max(np.abs(np.rad2deg(point.q - selected[-1].q)))
        )
        if (
            point.time - selected[-1].time >= minimum_period
            and joint_change_deg >= minimum_joint_change_deg
        ):
            selected.append(point)
    final_change_deg = float(
        np.max(np.abs(np.rad2deg(points[-1].q - selected[-1].q)))
    )
    if (
        len(points) > 1
        and points[-1] is not selected[-1]
        and final_change_deg >= minimum_joint_change_deg
    ):
        selected.append(points[-1])
    return selected


def build_jointmovj_dry_run(
    points: list[TrajectoryPoint],
    *,
    speed_percent: int = 10,
    acceleration_percent: int = 10,
    minimum_period: float = 0.25,
) -> list[str]:
    if not 1 <= speed_percent < 20:
        raise ValueError("real speed percent must be in [1, 20)")
    if not 1 <= acceleration_percent < 20:
        raise ValueError("real acceleration percent must be in [1, 20)")

    commands = []
    for point in downsample_points(points, minimum_period):
        joints_deg = sim_rad_to_real_deg(point.q)
        values = ",".join(f"{value:.6f}" for value in joints_deg)
        commands.append(
            f"JointMovJ({values},SpeedJ={speed_percent},"
            f"AccJ={acceleration_percent})"
        )
    return commands
