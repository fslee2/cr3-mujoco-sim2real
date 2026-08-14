"""Single source of truth for Dobot CR3 real/simulation joint conversion."""

from __future__ import annotations

import numpy as np


JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

# q_sim[i] = sign[i] * deg2rad(q_real[source_index[i]]) + offset_rad[i]
REAL_TO_SIM_SOURCE_INDEX = np.arange(6, dtype=int)
JOINT_SIGN = np.ones(6, dtype=float)
JOINT_OFFSET_RAD = np.zeros(6, dtype=float)

# Verified against the physical CR3 through DobotStudio Cartesian motion:
# J1..J6 order, direction, and zero convention match the MuJoCo model.
MAPPING_CALIBRATED = True


def _six_values(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (6,):
        raise ValueError(f"{name} must contain exactly 6 values, got {array.shape}")
    return array


def real_deg_to_sim_rad(joints_deg) -> np.ndarray:
    """Convert CR3 J1..J6 degrees to MuJoCo joint radians."""
    real = _six_values(joints_deg, "joints_deg")
    return (
        JOINT_SIGN * np.deg2rad(real[REAL_TO_SIM_SOURCE_INDEX])
        + JOINT_OFFSET_RAD
    )


def sim_rad_to_real_deg(joints_rad) -> np.ndarray:
    """Apply the exact inverse mapping from MuJoCo radians to CR3 degrees."""
    sim = _six_values(joints_rad, "joints_rad")
    real = np.empty(6, dtype=float)
    for sim_index, real_index in enumerate(REAL_TO_SIM_SOURCE_INDEX):
        real[real_index] = np.rad2deg(
            (sim[sim_index] - JOINT_OFFSET_RAD[sim_index])
            / JOINT_SIGN[sim_index]
        )
    return real


def mapping_metadata() -> dict:
    return {
        "joint_names": list(JOINT_NAMES),
        "real_to_sim_source_index": REAL_TO_SIM_SOURCE_INDEX.tolist(),
        "joint_sign": JOINT_SIGN.tolist(),
        "joint_offset_rad": JOINT_OFFSET_RAD.tolist(),
        "calibrated": MAPPING_CALIBRATED,
    }
