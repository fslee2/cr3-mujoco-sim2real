"""Dry-run command gate for a future guarded HaMeR-to-CR3 ServoJ path.

This prototype deliberately never opens a socket and never sends a robot
command. It converts a MuJoCo joint target into the exact rate-limited joint
command that a later real controller would be allowed to stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cr3_sim2real.hardware import (
    RobotFeedback,
    STRICT_20_PERCENT_LIMIT_DEG_S,
    limit_joint_velocity,
    require_motion_ready,
)
from cr3_sim2real.joint_mapping import sim_rad_to_real_deg


@dataclass(frozen=True)
class HamerRealSafetyConfig:
    max_joint_speed_deg_s: float = 5.0
    max_start_error_deg: float = 3.0
    max_tracking_error_deg: float = 5.0
    hand_watchdog_s: float = 0.50

    def validate(self) -> None:
        values = np.asarray(
            [
                self.max_joint_speed_deg_s,
                self.max_start_error_deg,
                self.max_tracking_error_deg,
                self.hand_watchdog_s,
            ],
            dtype=float,
        )
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("HaMeR real safety limits must be positive and finite")
        if self.max_joint_speed_deg_s >= STRICT_20_PERCENT_LIMIT_DEG_S:
            raise ValueError("HaMeR real speed must remain strictly below 36 deg/s")


@dataclass(frozen=True)
class HamerRealPlan:
    requested_deg: np.ndarray
    planned_deg: np.ndarray
    tracking_error_deg: float


class HamerRealCommandPreview:
    """Plan guarded CR3 commands without transmitting them."""

    def __init__(self, config: HamerRealSafetyConfig | None = None) -> None:
        self.config = config or HamerRealSafetyConfig()
        self.config.validate()
        self.armed = False
        self._last_planned_deg: np.ndarray | None = None
        self._last_plan_at = 0.0
        self._last_hand_update_at = 0.0

    def arm(self, sim_target_rad, feedback: RobotFeedback, now: float) -> float:
        require_motion_ready(feedback)
        requested = sim_rad_to_real_deg(sim_target_rad)
        start_error = float(np.max(np.abs(requested - feedback.joints_deg)))
        if start_error > self.config.max_start_error_deg:
            raise RuntimeError(
                f"HaMeR real start differs by {start_error:.3f} deg; "
                f"limit is {self.config.max_start_error_deg:.3f} deg"
            )
        self.armed = True
        self._last_planned_deg = feedback.joints_deg.copy()
        self._last_plan_at = float(now)
        self._last_hand_update_at = float(now)
        return start_error

    def disarm(self) -> None:
        self.armed = False
        self._last_planned_deg = None

    def plan(
        self,
        sim_target_rad,
        feedback: RobotFeedback,
        now: float,
    ) -> HamerRealPlan:
        if not self.armed or self._last_planned_deg is None:
            raise RuntimeError("HaMeR real command preview is not armed")
        require_motion_ready(feedback)
        tracking_error = float(
            np.max(np.abs(feedback.joints_deg - self._last_planned_deg))
        )
        dt = float(now) - self._last_plan_at
        if not np.isfinite(dt) or dt <= 0.0:
            raise RuntimeError("HaMeR command timestamps must increase")
        if dt > self.config.hand_watchdog_s:
            raise RuntimeError(f"HaMeR hand target is stale ({dt:.3f} s)")

        requested = sim_rad_to_real_deg(sim_target_rad)
        planned = limit_joint_velocity(
            self._last_planned_deg,
            requested,
            max_joint_speed_deg_s=self.config.max_joint_speed_deg_s,
            dt=dt,
        )
        self._last_planned_deg = planned.copy()
        self._last_plan_at = float(now)
        self._last_hand_update_at = float(now)
        return HamerRealPlan(
            requested_deg=requested,
            planned_deg=planned,
            tracking_error_deg=tracking_error,
        )

    def watchdog(self, feedback: RobotFeedback, now: float) -> None:
        if not self.armed or self._last_planned_deg is None:
            raise RuntimeError("HaMeR real command preview is not armed")
        require_motion_ready(feedback)
        age = float(now) - self._last_hand_update_at
        if age > self.config.hand_watchdog_s:
            raise RuntimeError(f"HaMeR hand target is stale ({age:.3f} s)")

    def require_tracking_within_limit(self, feedback: RobotFeedback) -> float:
        """Future senders must call this before transmitting a planned target."""
        if not self.armed or self._last_planned_deg is None:
            raise RuntimeError("HaMeR real command preview is not armed")
        require_motion_ready(feedback)
        tracking_error = float(
            np.max(np.abs(feedback.joints_deg - self._last_planned_deg))
        )
        if tracking_error > self.config.max_tracking_error_deg:
            raise RuntimeError(
                f"HaMeR real tracking error {tracking_error:.3f} deg exceeds "
                f"{self.config.max_tracking_error_deg:.3f} deg"
            )
        return tracking_error
