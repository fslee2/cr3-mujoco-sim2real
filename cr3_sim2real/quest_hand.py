"""Meta Quest Hand Tracking Streamer receiver and MuJoCo position mapper.

The socket lifecycle is adapted directly from the repository's
``scripts/sockets.py`` UDP listener and threaded TCP server.  The receiver
understands the exact UTF-8 CSV lines documented by that repository.  This
module deliberately contains no Dobot API calls: it only exposes fresh hand
telemetry and a relative end-effector position target for the GUI simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import socket
import threading
import time

import numpy as np


# Unity left-handed world (x right, y up, z forward) -> the MuJoCo/CR3 world
# convention used by this project (x forward, y left, z up).
R_UNITY_TO_ROBOT = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)

_FRAME_RE = re.compile(r"\bf\s*=\s*(\d+)", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(r"\bt\s*=\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class QuestPacket:
    side: str
    kind: str
    values: np.ndarray
    frame_id: int | None = None
    device_timestamp_ns: int | None = None


@dataclass(frozen=True)
class QuestHandSnapshot:
    side: str
    sequence: int
    wrist_sequence: int
    landmarks_sequence: int
    wrist_position: np.ndarray | None
    wrist_quaternion: np.ndarray | None
    landmarks: np.ndarray | None
    received_at: float
    sender: str
    packets_received: int

    @property
    def has_wrist(self) -> bool:
        # XYZ-only control does not need a wrist orientation.  Quaternion is
        # still retained in the snapshot for future orientation modes, but a
        # missing quaternion must not invalidate an otherwise good XYZ sample.
        return self.wrist_position is not None


def parse_quest_line(line: str) -> QuestPacket | None:
    """Parse one normal or debug-header HTS CSV line."""
    if ":" not in line:
        return None
    header, payload = line.split(":", 1)
    lower = header.lower()
    if "right" in lower:
        side = "right"
    elif "left" in lower:
        side = "left"
    elif "head" in lower:
        side = "head"
    else:
        return None

    if "landmarks" in lower:
        kind = "landmarks"
        expected = 63
    elif "wrist" in lower:
        kind = "wrist"
        expected = 7
    elif side == "head" and "pose" in lower:
        kind = "pose"
        expected = 7
    else:
        return None

    try:
        values = np.asarray(
            [float(part.strip()) for part in payload.split(",") if part.strip()],
            dtype=float,
        )
    except ValueError:
        return None
    if values.size < expected or not np.isfinite(values[:expected]).all():
        return None
    values = values[:expected].copy()
    if kind == "landmarks":
        values = values.reshape(21, 3)

    frame_match = _FRAME_RE.search(header)
    timestamp_match = _TIMESTAMP_RE.search(header)
    return QuestPacket(
        side=side,
        kind=kind,
        values=values,
        frame_id=int(frame_match.group(1)) if frame_match else None,
        device_timestamp_ns=(
            int(timestamp_match.group(1)) if timestamp_match else None
        ),
    )


class QuestHandReceiver:
    """Background UDP/TCP listener for Hand Tracking Streamer telemetry."""

    def __init__(self, protocol: str = "udp", host: str = "0.0.0.0", port: int = 9000):
        protocol = protocol.lower().strip()
        if protocol not in {"udp", "tcp"}:
            raise ValueError("Quest protocol must be udp or tcp")
        if not 1 <= int(port) <= 65535:
            raise ValueError("Quest port must be in 1..65535")
        self.protocol = protocol
        self.host = host.strip() or "0.0.0.0"
        self.port = int(port)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._clients: set[socket.socket] = set()
        self._sequence = 0
        self._packets_received = 0
        self._state = {
            "right": self._empty_state(),
            "left": self._empty_state(),
            "head": self._empty_state(),
        }
        self.status = "not started"

    @staticmethod
    def _empty_state() -> dict:
        return {
            "sequence": 0,
            "wrist_sequence": 0,
            "landmarks_sequence": 0,
            "wrist_position": None,
            "wrist_quaternion": None,
            "landmarks": None,
            "wrist_received_at": 0.0,
            "sender": "",
        }

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        listener_type = socket.SOCK_DGRAM if self.protocol == "udp" else socket.SOCK_STREAM
        listener = socket.socket(socket.AF_INET, listener_type)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.settimeout(0.25)
        if self.protocol == "tcp":
            listener.listen(5)
        self._socket = listener
        self.status = f"listening on {self.host}:{self.port}/{self.protocol}"
        target = self._run_udp if self.protocol == "udp" else self._run_tcp
        self._thread = threading.Thread(
            target=target,
            name=f"quest-{self.protocol}-receiver",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        listener = self._socket
        self._socket = None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            try:
                client.close()
            except OSError:
                pass
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self.status = "stopped"

    def get(self, side: str = "right") -> QuestHandSnapshot:
        side = side.lower().strip()
        if side not in self._state:
            raise ValueError(f"Unknown Quest side: {side}")
        with self._lock:
            state = self._state[side]
            return QuestHandSnapshot(
                side=side,
                sequence=int(state["sequence"]),
                wrist_sequence=int(state["wrist_sequence"]),
                landmarks_sequence=int(state["landmarks_sequence"]),
                wrist_position=(
                    None
                    if state["wrist_position"] is None
                    else state["wrist_position"].copy()
                ),
                wrist_quaternion=(
                    None
                    if state["wrist_quaternion"] is None
                    else state["wrist_quaternion"].copy()
                ),
                landmarks=(
                    None if state["landmarks"] is None else state["landmarks"].copy()
                ),
                received_at=float(state["wrist_received_at"]),
                sender=str(state["sender"]),
                packets_received=self._packets_received,
            )

    def _accept_text(self, text: str, sender: str) -> None:
        for line in text.splitlines():
            packet = parse_quest_line(line.strip())
            if packet is None:
                continue
            now = time.monotonic()
            with self._lock:
                state = self._state[packet.side]
                if packet.kind in {"wrist", "pose"}:
                    state["wrist_position"] = packet.values[:3].copy()
                    state["wrist_quaternion"] = packet.values[3:7].copy()
                    state["wrist_received_at"] = now
                    state["wrist_sequence"] += 1
                elif packet.kind == "landmarks":
                    state["landmarks"] = packet.values.copy()
                    state["landmarks_sequence"] += 1
                state["sender"] = sender
                self._sequence += 1
                state["sequence"] = self._sequence
                self._packets_received += 1
            self.status = f"receiving from {sender}"

    def _run_udp(self) -> None:
        listener = self._socket
        if listener is None:
            return
        while not self._stop.is_set():
            try:
                data, address = listener.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            self._accept_text(text, f"{address[0]}:{address[1]}")

    def _run_tcp(self) -> None:
        listener = self._socket
        if listener is None:
            return
        while not self._stop.is_set():
            try:
                client, address = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client.settimeout(0.25)
            with self._lock:
                self._clients.add(client)
            threading.Thread(
                target=self._handle_tcp_client,
                args=(client, f"{address[0]}:{address[1]}"),
                daemon=True,
                name="quest-tcp-client",
            ).start()

    def _handle_tcp_client(self, client: socket.socket, sender: str) -> None:
        buffer = ""
        try:
            while not self._stop.is_set():
                try:
                    data = client.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                try:
                    buffer += data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                lines = buffer.split("\n")
                buffer = lines.pop()
                self._accept_text("\n".join(lines), sender)
            if buffer.strip():
                self._accept_text(buffer, sender)
        finally:
            with self._lock:
                self._clients.discard(client)
            try:
                client.close()
            except OSError:
                pass


class QuestWristMapper:
    """Map an explicitly calibrated Quest wrist displacement to Link6 XYZ."""

    def __init__(
        self,
        gain: float = 1.0,
        max_delta_m: float = 0.20,
        deadzone_m: float = 0.001,
        ema_alpha: float = 0.30,
        fast_ema_alpha: float | None = None,
        slow_speed_m_s: float = 0.02,
        fast_speed_m_s: float = 0.20,
        filter_reference_hz: float = 60.0,
        # Position control is intentionally non-predictive by default.  A
        # predictive lead is useful for latency compensation, but it makes a
        # hand stop/reverse overshoot when UDP timing is irregular.  The GUI
        # can opt in explicitly if a different transport needs it.
        max_predict_s: float = 0.0,
        velocity_alpha: float = 0.80,
    ) -> None:
        self.gain = float(gain)
        self.max_delta_m = float(max_delta_m)
        self.deadzone_m = float(deadzone_m)
        self.ema_alpha = float(ema_alpha)
        self.fast_ema_alpha = float(
            max(self.ema_alpha, 0.85)
            if fast_ema_alpha is None
            else fast_ema_alpha
        )
        self.slow_speed_m_s = float(slow_speed_m_s)
        self.fast_speed_m_s = float(fast_speed_m_s)
        self.filter_reference_hz = float(filter_reference_hz)
        self.max_predict_s = float(max_predict_s)
        self.velocity_alpha = float(velocity_alpha)
        if not 0.0 < self.ema_alpha <= self.fast_ema_alpha <= 1.0:
            raise ValueError("Quest filter alpha must satisfy 0 < slow <= fast <= 1")
        if not 0.0 <= self.slow_speed_m_s < self.fast_speed_m_s:
            raise ValueError("Quest filter speed thresholds are invalid")
        if self.filter_reference_hz <= 0.0:
            raise ValueError("Quest filter reference rate must be positive")
        if self.max_predict_s < 0.0:
            raise ValueError("Quest prediction horizon must be non-negative")
        if not 0.0 < self.velocity_alpha <= 1.0:
            raise ValueError("Quest velocity alpha must satisfy 0 < alpha <= 1")
        self.wrist_origin: np.ndarray | None = None
        self.ee_origin: np.ndarray | None = None
        self._filtered: np.ndarray | None = None
        self._velocity = np.zeros(3)
        self._last_delta = np.zeros(3)
        self._last_sample_time: float | None = None
        self._last_filter_time: float | None = None
        self.last_wrist_speed_m_s = 0.0
        self.last_alpha = self.ema_alpha
        self.last_sample_hz = 0.0

    @property
    def calibrated(self) -> bool:
        return self.wrist_origin is not None and self.ee_origin is not None

    def clear_origin(self) -> None:
        self.wrist_origin = None
        self.ee_origin = None
        self._filtered = None
        self._velocity = np.zeros(3)
        self._last_delta = np.zeros(3)
        self._last_sample_time = None
        self._last_filter_time = None
        self.last_wrist_speed_m_s = 0.0
        self.last_alpha = self.ema_alpha
        self.last_sample_hz = 0.0

    def calibrate(
        self,
        wrist_position: np.ndarray,
        ee_position: np.ndarray,
        timestamp: float | None = None,
    ) -> None:
        wrist = np.asarray(wrist_position, dtype=float).reshape(3)
        ee = np.asarray(ee_position, dtype=float).reshape(3)
        if not np.isfinite(wrist).all() or not np.isfinite(ee).all():
            raise ValueError("Quest origin must contain finite values")
        self.wrist_origin = wrist.copy()
        self.ee_origin = ee.copy()
        self._filtered = ee.copy()
        self._velocity = np.zeros(3)
        self._last_delta = np.zeros(3)
        self._last_sample_time = (
            float(timestamp)
            if timestamp is not None and np.isfinite(timestamp)
            else None
        )
        self._last_filter_time = None
        self.last_wrist_speed_m_s = 0.0
        self.last_alpha = self.ema_alpha
        self.last_sample_hz = 0.0

    def reanchor_robot_origin(self, ee_position: np.ndarray) -> None:
        if self.wrist_origin is None:
            raise RuntimeError("Quest wrist origin has not been set")
        ee = np.asarray(ee_position, dtype=float).reshape(3)
        if not np.isfinite(ee).all():
            raise ValueError("Quest robot origin must contain finite values")
        self.ee_origin = ee.copy()
        self._filtered = ee.copy()
        # Re-anchoring (for example after Home) starts a new motion segment.
        # Do not carry velocity or packet timing from the previous segment into
        # the new origin, otherwise the first target can jump or drift.
        self._velocity = np.zeros(3)
        self._last_delta = np.zeros(3)
        self._last_sample_time = None
        self._last_filter_time = None
        self.last_wrist_speed_m_s = 0.0
        self.last_alpha = self.ema_alpha
        self.last_sample_hz = 0.0

    def target_pos(
        self,
        wrist_position: np.ndarray,
        timestamp: float | None = None,
        now: float | None = None,
    ) -> np.ndarray:
        if not self.calibrated:
            raise RuntimeError("Quest origin has not been set")
        assert self.wrist_origin is not None and self.ee_origin is not None
        wrist = np.asarray(wrist_position, dtype=float).reshape(3)
        relative = wrist - self.wrist_origin
        robot_delta = self.gain * (R_UNITY_TO_ROBOT @ relative)
        robot_delta[np.abs(robot_delta) < self.deadzone_m] = 0.0
        robot_delta = np.clip(robot_delta, -self.max_delta_m, self.max_delta_m)
        sample_time = (
            float(timestamp)
            if timestamp is not None and np.isfinite(timestamp)
            else time.monotonic()
        )
        eval_time = (
            float(now)
            if now is not None and np.isfinite(now)
            else sample_time
        )

        # Estimate velocity only on genuine new samples and smooth it, so
        # jittery packet intervals no longer feed noise into the target.
        if self._last_sample_time is None or sample_time > self._last_sample_time:
            if self._last_sample_time is not None:
                dt = sample_time - self._last_sample_time
                if dt > 0.0:
                    inst_vel = (robot_delta - self._last_delta) / dt
                    self.last_sample_hz = 1.0 / dt
                    self._velocity += self.velocity_alpha * (
                        inst_vel - self._velocity
                    )
                    self.last_wrist_speed_m_s = float(
                        np.linalg.norm(self._velocity)
                    )
            self._last_delta = robot_delta.copy()
            self._last_sample_time = sample_time

        # Extrapolate the newest sample forward to the evaluation time so the
        # control loop keeps moving smoothly between packets instead of
        # freezing at each sample and stepping on the next one.
        if (
            self._last_sample_time is not None
            and eval_time > self._last_sample_time
        ):
            horizon = min(
                eval_time - self._last_sample_time, self.max_predict_s
            )
            predicted_delta = self._last_delta + self._velocity * horizon
        else:
            predicted_delta = self._last_delta

        raw_target = self.ee_origin + predicted_delta

        speed_blend = np.clip(
            (self.last_wrist_speed_m_s - self.slow_speed_m_s)
            / (self.fast_speed_m_s - self.slow_speed_m_s),
            0.0,
            1.0,
        )
        base_alpha = float(
            self.ema_alpha
            + speed_blend * (self.fast_ema_alpha - self.ema_alpha)
        )
        if self._last_filter_time is None:
            alpha = base_alpha
        else:
            filter_dt = max(eval_time - self._last_filter_time, 0.0)
            reference_steps = float(
                np.clip(filter_dt * self.filter_reference_hz, 0.25, 4.0)
            )
            alpha = 1.0 - (1.0 - base_alpha) ** reference_steps
        self.last_alpha = float(np.clip(alpha, 0.0, 1.0))
        if self._filtered is None:
            self._filtered = raw_target.copy()
        else:
            self._filtered += self.last_alpha * (raw_target - self._filtered)
        self._last_filter_time = eval_time
        return self._filtered.copy()
