"""Small HaMeR HTTP bridge client and relative arm-position mapper.

This module intentionally has no Dobot dependency.  It produces a MuJoCo
end-effector target only; deciding whether a target may later be mirrored to
hardware belongs to the guarded GUI/controller layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import platform
import threading
import time
from typing import Any
import urllib.error
import urllib.request

import cv2
import numpy as np


# Camera (x right, y down, z depth) -> CR3/MuJoCo world
# (x forward, y left, z up), for a camera facing the robot.
R_CAM_TO_ROBOT = np.array(
    [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)


def hamer_cam_t(result: dict[str, Any] | None) -> np.ndarray | None:
    """Return a validated HaMeR camera translation, or ``None``."""
    if not result or "cam_t" not in result:
        return None
    try:
        value = np.asarray(result["cam_t"], dtype=float).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value).all():
        return None
    return value


class PositionFilter:
    def __init__(
        self,
        alpha: float = 0.15,
        deadzone: float = 0.03,
        max_step: float | None = 0.06,
    ) -> None:
        self.alpha = float(alpha)
        self.deadzone = float(deadzone)
        self.max_step = None if max_step is None else float(max_step)
        self.previous: np.ndarray | None = None

    def reset(self) -> None:
        self.previous = None

    def apply(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=float).reshape(3)
        if self.previous is None:
            self.previous = target.copy()
            return target.copy()
        delta = target - self.previous
        if float(np.linalg.norm(delta)) < self.deadzone:
            return self.previous.copy()
        filtered = self.previous + self.alpha * delta
        step = filtered - self.previous
        norm = float(np.linalg.norm(step))
        if self.max_step is not None and norm > self.max_step:
            filtered = self.previous + step / norm * self.max_step
        self.previous = filtered.copy()
        return filtered


class HamerArmMapper:
    """Map relative HaMeR ``cam_t`` motion to a MuJoCo Link6 position.

    Calibration is deliberately explicit: ``target_pos`` raises until
    ``calibrate`` has been called by a user action.  No first-frame motion is
    ever accepted as an implicit origin.
    """

    def __init__(
        self,
        depth_scale: float = 0.03,
        lateral_vertical_scale: float = 2.0,
        depth_deadband: float = 0.015,
        max_delta: float | np.ndarray = 0.15,
        ema_alpha: float = 0.15,
        filter_deadzone: float = 0.03,
        filter_max_step: float | None = 0.06,
    ) -> None:
        self.scale = np.array(
            [depth_scale, lateral_vertical_scale, lateral_vertical_scale],
            dtype=float,
        )
        self.depth_deadband = float(depth_deadband)
        max_delta_array = np.asarray(max_delta, dtype=float)
        if max_delta_array.ndim == 0:
            max_delta_array = np.full(3, float(max_delta_array))
        self.max_delta = max_delta_array.reshape(3)
        self.filter = PositionFilter(
            alpha=ema_alpha,
            deadzone=filter_deadzone,
            max_step=filter_max_step,
        )
        self.cam_origin: np.ndarray | None = None
        self.ee_origin: np.ndarray | None = None

    @property
    def calibrated(self) -> bool:
        return self.cam_origin is not None and self.ee_origin is not None

    def clear_origin(self) -> None:
        self.cam_origin = None
        self.ee_origin = None
        self.filter.reset()

    def calibrate(self, cam_t: np.ndarray, ee_pos: np.ndarray) -> None:
        cam_t = np.asarray(cam_t, dtype=float).reshape(3)
        ee_pos = np.asarray(ee_pos, dtype=float).reshape(3)
        if not np.isfinite(cam_t).all() or not np.isfinite(ee_pos).all():
            raise ValueError("HaMeR origin must contain finite values")
        self.cam_origin = cam_t.copy()
        self.ee_origin = ee_pos.copy()
        self.filter.reset()

    def reanchor_robot_origin(self, ee_pos: np.ndarray) -> None:
        """Keep the hand origin but move its robot-side anchor to ``ee_pos``."""
        if self.cam_origin is None:
            raise RuntimeError("HaMeR hand origin has not been set")
        ee_pos = np.asarray(ee_pos, dtype=float).reshape(3)
        if not np.isfinite(ee_pos).all():
            raise ValueError("HaMeR robot origin must contain finite values")
        self.ee_origin = ee_pos.copy()
        self.filter.reset()

    def target_pos(self, cam_t: np.ndarray) -> np.ndarray:
        if not self.calibrated:
            raise RuntimeError("HaMeR origin has not been set")
        assert self.cam_origin is not None and self.ee_origin is not None
        relative = np.asarray(cam_t, dtype=float).reshape(3) - self.cam_origin
        if abs(relative[2]) < self.depth_deadband:
            relative[2] = 0.0
        delta = self.scale * (R_CAM_TO_ROBOT @ relative)
        delta = np.clip(delta, -self.max_delta, self.max_delta)
        return self.filter.apply(self.ee_origin + delta)


@dataclass(frozen=True)
class BridgeSnapshot:
    sequence: int
    result: dict[str, Any] | None
    frame_bgr: np.ndarray | None
    status: str
    frame_sequence: int = 0
    request_latency_s: float = 0.0
    request_in_flight: bool = False
    result_at: float = 0.0


class HamerBridgeTracker:
    """Keep camera display live while HaMeR processes the newest full frame."""

    def __init__(
        self,
        bridge_url: str,
        *,
        video_path: str | Path | None = None,
        camera_id: int = 1,
        camera_backend: str = "auto",
        frame_stride: int = 1,
        jpeg_quality: int = 80,
        request_timeout: float = 30.0,
    ) -> None:
        url = str(bridge_url).strip().rstrip("/")
        if not url:
            raise ValueError("HaMeR bridge URL is empty")
        self.bridge_url = url if url.endswith("/infer") else url + "/infer"
        self.video_path = None if not video_path else Path(video_path)
        self.camera_id = int(camera_id)
        self.camera_backend = str(camera_backend).lower()
        self.frame_stride = max(1, int(frame_stride))
        self.jpeg_quality = int(np.clip(jpeg_quality, 20, 100))
        self.request_timeout = float(request_timeout)

        self._lock = threading.Lock()
        self._running = False
        self._capture_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._inference_ready = threading.Event()
        self._capture = None
        self._sequence = 0
        self._frame_sequence = 0
        self._inference_frame_sequence = 0
        self._inference_frame: np.ndarray | None = None
        self._capture_ended = False
        self._result: dict[str, Any] | None = None
        self._frame: np.ndarray | None = None
        self._status = "HaMeR bridge not started"
        self._request_latency_s = 0.0
        self._request_in_flight = False
        self._result_at = 0.0

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        if self.video_path is not None:
            if not self.video_path.is_file():
                raise FileNotFoundError(f"HaMeR video was not found: {self.video_path}")
            capture = cv2.VideoCapture(str(self.video_path))
        else:
            capture = self._open_camera()
        if not capture.isOpened():
            source = self.video_path if self.video_path is not None else f"camera {self.camera_id}"
            raise RuntimeError(f"Could not open HaMeR source: {source}")
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        self._capture_ended = False
        self._inference_ready.clear()
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="hamer-capture",
            daemon=True,
        )
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="hamer-inference",
            daemon=True,
        )
        self._capture_thread.start()
        self._inference_thread.start()

    def _open_camera(self):
        backend = self.camera_backend
        if backend == "auto":
            backend = "dshow" if platform.system().lower().startswith("win") else "any"
        backends = {
            "any": None,
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "v4l2": cv2.CAP_V4L2,
        }
        if backend not in backends:
            raise ValueError("camera backend must be auto, any, dshow, msmf, or v4l2")
        selected = backends[backend]
        return (
            cv2.VideoCapture(self.camera_id)
            if selected is None
            else cv2.VideoCapture(self.camera_id, selected)
        )

    def _capture_loop(self) -> None:
        assert self._capture is not None
        frame_index = 0
        try:
            while self._running:
                ok, frame = self._capture.read()
                if not ok:
                    status = (
                        "HaMeR video ended"
                        if self.video_path is not None
                        else "HaMeR camera has no frames"
                    )
                    with self._lock:
                        self._status = status
                        if self.video_path is not None:
                            self._capture_ended = True
                    if self.video_path is not None:
                        self._inference_ready.set()
                        break
                    time.sleep(0.05)
                    continue
                frame = cv2.flip(frame, 1)
                frame_index += 1
                selected = frame_index % self.frame_stride == 0
                with self._lock:
                    self._frame_sequence += 1
                    self._frame = frame.copy()
                    if selected:
                        self._inference_frame_sequence = self._frame_sequence
                        self._inference_frame = frame.copy()
                if selected:
                    self._inference_ready.set()
        finally:
            with self._lock:
                self._capture_ended = True
            self._inference_ready.set()

    def _inference_loop(self) -> None:
        # These match the original working demo exactly: mirrored once by the
        # capture loop, no resize, and JPEG quality 80 by default.
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        consumed_sequence = 0
        try:
            while self._running:
                self._inference_ready.wait(timeout=0.1)
                self._inference_ready.clear()
                with self._lock:
                    frame_sequence = self._inference_frame_sequence
                    frame = (
                        None
                        if self._inference_frame is None
                        else self._inference_frame.copy()
                    )
                    capture_ended = self._capture_ended
                if frame is None or frame_sequence == consumed_sequence:
                    if capture_ended:
                        break
                    continue
                consumed_sequence = frame_sequence

                result = None
                status = self._status
                with self._lock:
                    self._request_in_flight = True
                request_started_at = time.monotonic()
                try:
                    ok, jpg = cv2.imencode(".jpg", frame, encode_params)
                    if not ok:
                        raise RuntimeError("could not encode HaMeR frame")
                    request = urllib.request.Request(
                        self.bridge_url,
                        data=jpg.tobytes(),
                        headers={"Content-Type": "image/jpeg"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    status = str(payload.get("status", "HaMeR bridge returned no status"))
                    if payload.get("ok", False):
                        candidate = payload.get("result")
                        result = candidate if isinstance(candidate, dict) else None
                except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                    status = f"HaMeR bridge error: {exc}"

                request_latency = time.monotonic() - request_started_at
                with self._lock:
                    self._sequence += 1
                    self._result = result
                    self._status = status
                    self._request_latency_s = request_latency
                    self._request_in_flight = False
                    self._result_at = time.monotonic()
                    capture_ended = self._capture_ended
                    newest_sequence = self._inference_frame_sequence
                if capture_ended and newest_sequence == consumed_sequence:
                    break
        finally:
            with self._lock:
                self._request_in_flight = False
            self._running = False

    def get(self) -> BridgeSnapshot:
        with self._lock:
            sequence = self._sequence
            result = None if self._result is None else dict(self._result)
            frame = None if self._frame is None else self._frame.copy()
            status = self._status
            frame_sequence = self._frame_sequence
            request_latency_s = self._request_latency_s
            request_in_flight = self._request_in_flight
            result_at = self._result_at
        if frame is not None:
            color = (
                (60, 220, 100)
                if hamer_cam_t(result) is not None
                else (40, 80, 240)
            )
            display_status = (
                f"infer pending | last: {status}"
                if request_in_flight
                else status
            )
            cv2.putText(
                frame,
                display_status[:100],
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
            )
        return BridgeSnapshot(
            sequence=sequence,
            result=result,
            frame_bgr=frame,
            status=status,
            frame_sequence=frame_sequence,
            request_latency_s=request_latency_s,
            request_in_flight=request_in_flight,
            result_at=result_at,
        )

    def stop(self) -> None:
        self._running = False
        self._inference_ready.set()
        if self._capture is not None:
            self._capture.release()
        current = threading.current_thread()
        for thread in (self._capture_thread, self._inference_thread):
            if thread is not None and thread is not current:
                thread.join(timeout=1.5)
        self._capture_thread = None
        self._inference_thread = None
        self._capture = None
