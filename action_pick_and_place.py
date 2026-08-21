#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文:
演示 CR3 机械臂完成一次"抓取-放置"动作:
  1) 从初始位姿移动到抓取位并短暂稳定;
  2) 到达抓取位后保持 2 秒, 触发灵巧手执行"抓住"(15 电机);
  3) 抬升并转运到放置位;
  4) 到达放置位后触发灵巧手执行"四指张开"释放物体, 保持结束位姿。

轨迹来自 trajectories/pickandplace.json。已精简(线性插值误差<0.5度):
原始的 15 秒"抓住"停顿压缩为 6 秒, 结束持位压缩为 5 秒; 手部触发点:
抓取位后 2 秒执行"抓住", 结束持位执行"四指张开"。

English:
Demonstrates a pick-and-place routine on a CR3 arm:
  1) move to the grasp pose and settle briefly;
  2) hold at the grasp pose for 2 s, then trigger the dexterous hand "grab";
  3) lift and transport to the place pose;
  4) at the place pose trigger the hand "four-finger open" to release.

Source: trajectories/pickandplace.json (simplified, <0.5 deg interp error).
The 15 s grab hold was shortened to 6 s and the final hold to 5 s.

安全 / Safety:
默认 dry-run。真正运动需 --enable-real-execution 并输入 YES。程序会读取反馈端口
确认机器人使能/无报警/可运动，从当前位姿缓坡到起点，33 Hz ServoJ 限速流式播放。
手部: --hand auto(默认) 直接驱动 CRAFT Hand 串口到 15 电机目标;
--hand manual 则打印提示等回车。
请确保工作区安全。Dry-run by default. Real motion needs --enable-real-execution
plus a typed YES. It reads the feedback port, ramps to the start pose, and streams
velocity-limited ServoJ at 33 Hz. Hand: --hand auto (default) drives the CRAFT Hand
over serial to the 15-motor target; --hand manual prints the target and waits for Enter.
Keep the workspace clear."""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import threading
import time

import numpy as np

from dobot_api import DobotApiDashboard, DobotApiMove, MyType

# --------------------------------------------------------------------------
# 端口与机器人参数 (ports & robot)
# --------------------------------------------------------------------------
DEFAULT_ROBOT_IP = "192.168.5.11"
DASHBOARD_PORT = 29999          # 控制指令端口
MOTION_PORT = 30003             # 运动指令端口 (ServoJ)
FEEDBACK_PORT = 30004           # 反馈端口 (关节状态)
PACKET_MARKER = 0x123456789ABCDEF  # 反馈包帧对齐标记

# --------------------------------------------------------------------------
# ServoJ 流参数 (streaming parameters)
# --------------------------------------------------------------------------
SERVO_PERIOD_S = 0.03          # 33 Hz 控制周期
SERVO_T_S = 0.10               # ServoJ 插值窗口
LOOKAHEAD_TIME = 50
GAIN = 500
RAMP_DURATION_S = 3.0          # 从当前位姿缓坡到轨迹起点的时长

MAX_JOINT_SPEED_DEG_S = 80.0     # 不低于录制峰值速度(69 deg/s), 避免失真
GRAB_TRIGGER_T = 15.578  # 进入抓取停顿 2 s 后触发手部"抓住"
OPEN_TRIGGER_T = 27.844  # 到达结束持位时触发手部"四指张开"
# 手部动作: 15 个电机目标位置 (rad)
# Hand actions: 15 motor target positions (rad), from
# trajectories/craft_hand_positions/抓住.json & 四指张开.json
HAND_POSITIONS = {
    "grab": [5.5315, 2.9974, 4.3412, 6.1635, 3.6003, 6.8277, 4.2399, 0.3682, 5.5300, 3.5082, 3.3195, 0.2224, 2.3286, 3.2536, 3.2183],
    "open": [5.4886, 4.5314, 4.7431, 6.1635, 4.8320, 6.8354, 4.2369, 1.0492, 5.0974, 3.5190, 4.5866, 0.2577, 2.3286, 4.5590, 4.3258],
}
# HAND_DRIVER: 手部自动下发扩展点。默认 None=手动(打印提示等回车)。
# --hand auto 时 main() 用 CraftHandDriver 直接驱动串口到 15 电机目标(rad)。
# 也可自行赋值为 lambda motor_ids, positions_rad: <你的下发逻辑>。
# Extension point for automatic hand control; None = manual prompt + Enter.
HAND_DRIVER = None


class CraftHandDriver:
    """直接驱动 CRAFT Hand 到 15 电机目标位置 (rad)。

    Dynamixel 串口直连 (默认 COM11 @ 57600)。按项目安全配置初始化:
    电流模式位置控制 + 限流 + 位置闭环 PID。write_pose() 一次下发 15 个
    电机目标, 电机以受限电流平滑移动到目标。

    Directly drives the CRAFT Hand to a 15-motor target (rad) over the
    Dynamixel serial bus using the project's safe config (current-limited
    position control). Requires CRAFT-Hand_API and dynamixel_sdk.
    """

    MOTOR_IDS = list(range(1, 16))

    def __init__(self, port="COM12", baud=57600, current_ma=350):
        from pathlib import Path

        try:
            from craft_hand_utils.dynamixel_client import (
                ADDR_GOAL_CURRENT,
                ADDR_OPERATING_MODE,
                ADDR_POSITION_D_GAIN,
                ADDR_POSITION_I_GAIN,
                ADDR_POSITION_P_GAIN,
                DynamixelClient,
                LEN_GOAL_CURRENT,
                LEN_OPERATING_MODE,
                LEN_PID_GAIN,
            )
        except ImportError:
            here = Path(__file__).resolve()
            for cand in (
                here.parent.parent.parent / "CRAFT-Hand_API" / "python",
                here.parent / "craft_hand_utils",
            ):
                if cand.exists():
                    if str(cand) not in sys.path:
                        sys.path.insert(0, str(cand))
                    break
            try:
                from craft_hand_utils.dynamixel_client import (
                    ADDR_GOAL_CURRENT,
                    ADDR_OPERATING_MODE,
                    ADDR_POSITION_D_GAIN,
                    ADDR_POSITION_I_GAIN,
                    ADDR_POSITION_P_GAIN,
                    DynamixelClient,
                    LEN_GOAL_CURRENT,
                    LEN_OPERATING_MODE,
                    LEN_PID_GAIN,
                )
            except ImportError:
                raise RuntimeError(
                    "无法导入 craft_hand_utils (需要 CRAFT-Hand_API/python)。"
                    "请确认仓库存在, 或用 --hand manual 手动执行手部动作。\n"
                    "Cannot import craft_hand_utils; check CRAFT-Hand_API/python "
                    "or use --hand manual."
                )

        ids = self.MOTOR_IDS
        self._port = port
        self._baud = baud
        self._DynamixelClient = DynamixelClient
        self._client = DynamixelClient(ids, port, baud)
        self._client.connect()
        self._client.set_torque_enabled(ids, False, retries=3, retry_interval=0.05)
        time.sleep(0.05)
        # 电流模式位置控制 + PID + 限流 (与 CRAFT-Hand_API 安全配置一致)
        self._client.sync_write(ids, np.full(15, 5, dtype=np.int32),
                                ADDR_OPERATING_MODE, LEN_OPERATING_MODE)
        self._client.sync_write(ids, np.full(15, 400, dtype=np.int32),
                                ADDR_POSITION_P_GAIN, LEN_PID_GAIN)
        self._client.sync_write(ids, np.zeros(15, dtype=np.int32),
                                ADDR_POSITION_I_GAIN, LEN_PID_GAIN)
        self._client.sync_write(ids, np.zeros(15, dtype=np.int32),
                                ADDR_POSITION_D_GAIN, LEN_PID_GAIN)
        self._client.sync_write(ids, np.full(15, int(current_ma), dtype=np.int32),
                                ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT)
        current = self._client.read_pos()
        self._client.write_desired_pos(ids, current)
        self._client.set_torque_enabled(ids, True, retries=5, retry_interval=0.05)
        print("[hand] CRAFT Hand 已连接并使能: %s @ %d baud, 限流 %d mA"
              % (port, baud, int(current_ma)))

    PRESENT_POSITION_ADDR = 132  # Dynamixel Protocol2 控制表: 当前位置

    def keepalive(self):
        """单电机轻读, 保持串口活跃, 防止 USB 空闲挂起后写入被拒绝。"""
        try:
            self._client.read_bytes([1], self.PRESENT_POSITION_ADDR, 4)
        except Exception:
            pass

    def write_pose(self, motor_ids, positions_rad):
        """下发 15 电机目标位置 (rad)。Sends a 15-motor target pose (rad)."""
        target = np.asarray(positions_rad, dtype=float)
        try:
            self._client.write_desired_pos(self.MOTOR_IDS, target)
        except Exception as exc:
            print("[hand] 写入失败(%s), 重连串口后重试..." % exc)
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = self._DynamixelClient(self.MOTOR_IDS,
                                                 self._port, self._baud)
            self._client.connect()
            self._client.set_torque_enabled(self.MOTOR_IDS, True,
                                            retries=3, retry_interval=0.05)
            self._client.write_desired_pos(self.MOTOR_IDS, target)

    def close(self):
        try:
            self._client.set_torque_enabled(self.MOTOR_IDS, False,
                                            retries=3, retry_interval=0.05)
            self._client.disconnect()
            print("[hand] CRAFT Hand 已下使能并断开。")
        except Exception as exc:
            print("[hand] 关闭手部连接失败:", exc)

# --------------------------------------------------------------------------
# 内嵌轨迹点: (时间 s, [J1..J6] 度)
# Embedded waypoints: (time s, [J1..J6] deg)
# 来源: trajectories/pickandplace.json (已精简, 线性插值误差 < 0.5 度)
# --------------------------------------------------------------------------
WAYPOINTS_DEG = [
    (0.000, [75.421, -57.041, 46.756, -13.402, 173.488, 0.509]),
    (0.813, [75.421, -57.046, 46.565, -13.416, 173.488, 0.509]),
    (0.922, [75.421, -57.046, 45.644, -13.416, 173.488, 0.509]),
    (1.141, [75.421, -57.046, 39.981, -13.416, 173.488, 0.509]),
    (1.344, [75.421, -57.046, 32.103, -13.416, 173.488, 0.509]),
    (1.563, [75.427, -57.047, 24.119, -13.415, 173.488, 0.509]),
    (1.656, [75.503, -57.048, 20.495, -13.415, 173.488, 0.509]),
    (1.922, [76.416, -57.058, 12.511, -13.407, 173.488, 0.509]),
    (2.125, [77.487, -57.218, 6.690, -13.401, 173.488, 0.509]),
    (2.328, [77.492, -57.881, 0.888, -13.398, 173.488, 0.509]),
    (2.547, [77.493, -57.882, -5.550, -13.391, 173.489, 0.509]),
    (2.766, [77.494, -57.882, -12.690, -13.389, 173.489, 0.509]),
    (2.985, [77.443, -57.883, -20.670, -12.880, 173.489, 0.509]),
    (3.078, [77.328, -57.883, -23.673, -11.094, 173.489, 0.509]),
    (3.188, [77.168, -57.882, -26.316, -7.578, 173.488, 0.509]),
    (3.391, [77.177, -57.883, -27.661, 0.086, 173.489, 0.509]),
    (3.531, [77.157, -57.903, -31.030, 6.291, 173.489, 0.509]),
    (3.641, [77.093, -58.766, -35.873, 12.671, 173.489, 0.509]),
    (3.750, [76.872, -60.159, -39.573, 18.914, 173.488, 0.509]),
    (3.860, [76.342, -61.692, -43.728, 26.534, 173.488, 0.509]),
    (3.985, [75.587, -62.533, -47.724, 32.800, 173.487, 0.509]),
    (4.094, [74.861, -63.036, -51.316, 36.976, 173.485, 0.509]),
    (4.203, [74.600, -63.042, -54.677, 38.946, 173.484, 0.509]),
    (4.438, [74.600, -63.042, -59.667, 39.536, 173.484, 0.509]),
    (4.641, [74.594, -63.042, -62.683, 39.582, 173.486, 0.509]),
    (4.750, [74.594, -63.043, -64.163, 40.517, 173.487, 0.509]),
    (4.860, [74.594, -63.043, -65.647, 43.133, 173.487, 0.509]),
    (4.969, [74.574, -63.043, -67.191, 46.751, 173.486, 0.509]),
    (5.078, [74.441, -63.043, -68.436, 48.883, 173.484, 0.509]),
    (5.391, [74.328, -63.042, -69.014, 49.377, 173.476, 0.509]),
    (5.828, [74.334, -63.043, -68.766, 49.379, 173.477, 0.509]),
    (5.938, [74.335, -63.153, -66.958, 49.378, 173.476, 0.509]),
    (6.047, [74.334, -64.118, -63.955, 49.378, 173.476, 0.509]),
    (6.250, [74.376, -71.060, -56.195, 49.382, 173.476, 0.509]),
    (6.375, [74.382, -74.837, -51.660, 49.380, 173.476, 0.509]),
    (6.516, [74.390, -77.222, -47.900, 49.378, 173.476, 0.509]),
    (6.625, [74.347, -77.913, -47.086, 49.184, 173.473, 0.509]),
    (6.938, [74.346, -77.947, -48.922, 45.830, 173.473, 0.509]),
    (7.485, [74.180, -78.591, -49.312, 45.624, 173.473, 0.509]),
    (8.031, [74.020, -79.222, -48.522, 45.962, 173.471, 0.509]),
    (8.360, [74.029, -81.951, -48.521, 44.776, 173.471, 0.509]),
    (8.938, [73.213, -81.991, -48.516, 44.782, 173.471, 0.509]),
    (9.735, [69.398, -81.991, -48.519, 44.784, 173.470, 0.509]),
    (11.625, [69.625, -81.992, -48.560, 44.781, 173.149, 0.509]),
    (11.750, [69.741, -81.992, -48.560, 44.782, 172.389, 0.509]),
    (11.938, [70.611, -81.991, -48.557, 44.787, 168.484, 0.509]),
    (12.375, [71.760, -81.991, -48.556, 44.787, 164.218, 0.509]),
    (12.719, [71.759, -81.991, -48.556, 44.787, 163.258, 0.509]),
    (13.344, [71.757, -81.991, -48.556, 44.787, 162.787, 0.509]),
    (13.578, [71.757, -81.990, -48.036, 44.838, 158.728, 0.509]),
    (19.578, [71.750, -81.988, -47.482, 44.981, 158.495, 0.509]),
    (19.813, [71.750, -81.988, -44.488, 48.089, 158.504, 0.509]),
    (20.031, [71.750, -81.988, -43.178, 50.062, 158.504, 0.509]),
    (20.375, [71.746, -81.961, -38.154, 50.184, 158.504, 0.509]),
    (20.609, [70.985, -81.904, -32.561, 50.179, 158.504, 0.509]),
    (20.813, [68.447, -81.726, -26.740, 50.178, 158.503, 0.509]),
    (21.047, [64.396, -80.416, -19.813, 50.178, 158.503, 0.509]),
    (21.172, [61.680, -80.416, -17.286, 50.178, 158.503, 0.509]),
    (21.578, [54.103, -80.416, -12.937, 50.178, 158.503, 0.509]),
    (22.000, [46.229, -80.417, -12.517, 50.164, 158.502, 0.509]),
    (22.219, [42.707, -80.420, -13.041, 50.147, 158.502, 0.509]),
    (22.375, [40.940, -80.420, -15.706, 50.146, 158.501, 0.509]),
    (22.609, [39.668, -80.420, -20.743, 50.137, 158.498, 0.509]),
    (22.828, [38.486, -80.420, -22.833, 49.442, 158.493, 0.509]),
    (23.063, [37.156, -80.420, -23.573, 41.664, 158.493, 0.509]),
    (23.172, [37.003, -80.420, -23.573, 37.277, 158.494, 0.509]),
    (23.328, [37.001, -80.420, -23.572, 32.320, 158.494, 0.509]),
    (23.453, [37.001, -80.420, -23.572, 29.951, 158.494, 0.509]),
    (23.672, [36.855, -80.420, -23.573, 28.759, 158.494, 0.509]),
    (24.234, [35.317, -81.060, -23.577, 28.758, 158.493, 0.509]),
    (25.594, [34.512, -81.074, -23.618, 28.750, 158.337, 0.510]),
    (25.703, [34.388, -81.074, -23.620, 28.749, 156.608, 0.510]),
    (25.828, [34.390, -81.074, -23.620, 28.749, 152.676, 0.510]),
    (25.922, [34.391, -81.074, -23.620, 28.750, 147.955, 0.510]),
    (26.047, [34.393, -81.074, -23.620, 28.750, 145.096, 0.509]),
    (26.156, [34.425, -81.074, -23.620, 28.750, 145.663, 0.510]),
    (26.250, [34.425, -81.074, -23.620, 28.750, 148.201, 0.510]),
    (26.375, [34.425, -81.075, -23.620, 28.750, 152.879, 0.510]),
    (26.594, [34.505, -81.074, -24.829, 28.750, 159.033, 0.510]),
    (26.813, [35.138, -81.074, -29.434, 28.757, 164.152, 0.509]),
    (27.031, [34.343, -81.073, -29.749, 28.757, 165.081, 0.510]),
    (27.844, [33.376, -81.073, -29.748, 28.761, 165.081, 0.509]),
    (32.844, [33.378, -81.071, -29.695, 28.786, 165.081, 0.509]),
]

# --------------------------------------------------------------------------
# 反馈与状态 (feedback & state)
# --------------------------------------------------------------------------
def require_command_success(command, reply):
    """校验指令返回 ErrorID==0，否则抛出。Raise on a non-zero ErrorID reply."""
    text = (reply or "").strip()
    match = re.match(r"\s*(-?\d+)\s*,", text)
    if match is None:
        raise RuntimeError("%s returned an unrecognized reply: %r" % (command, text))
    error_id = int(match.group(1))
    if error_id != 0:
        raise RuntimeError("%s rejected by CR3 (ErrorID %d): %s" % (command, error_id, text))


def read_current_joints(ip, timeout_s=3.0):
    """从反馈端口读取当前关节角(度)与状态。失败返回 None。

    Reads live joint angles (deg) and status from the feedback port.
    Returns a dict, or None if the feedback stream is unavailable.
    """
    marker = PACKET_MARKER.to_bytes(8, byteorder="little")
    marker_offset = MyType.fields["test_value"][1]
    packet_size = MyType.itemsize
    try:
        with socket.create_connection((ip, FEEDBACK_PORT), timeout=timeout_s) as sock:
            sock.settimeout(0.5)
            buffer = bytearray()
            deadline = time.monotonic() + timeout_s * 2.0
            while time.monotonic() < deadline:
                try:
                    chunk = sock.recv(packet_size * 10)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buffer.extend(chunk)
                while True:
                    index = buffer.find(marker)
                    if index < 0:
                        if len(buffer) > packet_size * 2:
                            del buffer[:-packet_size]
                        break
                    start = index - marker_offset
                    if start < 0:
                        del buffer[: index + len(marker)]
                        continue
                    end = start + packet_size
                    if len(buffer) < end:
                        if start > 0:
                            del buffer[:start]
                        break
                    raw = bytes(buffer[start:end])
                    del buffer[:end]
                    parsed = np.frombuffer(raw, dtype=MyType, count=1)
                    return {
                        "joints_deg": parsed["q_actual"][0].copy(),
                        "robot_mode": int(parsed["robot_mode"][0]),
                        "enabled": bool(parsed["enable_status"][0]),
                        "error": bool(parsed["error_status"][0]),
                        "queue_running": bool(parsed["run_queued_cmd"][0]),
                    }
            return None
    except OSError:
        return None


def require_motion_ready(state):
    """安全门: 使能、无报警、可运动模式。Safety gate before any motion."""
    if state is None:
        raise RuntimeError("无法读取反馈端口 %d，不能确认机器人状态，已中止" % FEEDBACK_PORT)
    if state["error"] or state["robot_mode"] == 9:
        raise RuntimeError("CR3 有报警/错误，已中止")
    if not state["enabled"]:
        raise RuntimeError("CR3 未使能；请先在 DobotStudio 中使能")
    if state["robot_mode"] not in (5, 7):
        raise RuntimeError("CR3 模式 %d 不是就绪/运行状态" % state["robot_mode"])


def connect(ip):
    """连接 dashboard + motion 端口并使能。Connect and enable the robot."""
    dashboard = DobotApiDashboard(ip, DASHBOARD_PORT)
    move = DobotApiMove(ip, MOTION_PORT)
    require_command_success("EnableRobot", dashboard.EnableRobot())
    return dashboard, move


def ensure_queue_running(dashboard, state):
    """若命令队列停止则发送 Continue()。Resume a stopped command queue."""
    if state["queue_running"]:
        return
    print("[robot] 命令队列已停止，发送 Continue()")
    require_command_success("Continue", dashboard.Continue())


# --------------------------------------------------------------------------
# ServoJ 流 (streaming)
# --------------------------------------------------------------------------
def resample_stream(waypoints, period=SERVO_PERIOD_S):
    """把内嵌路径点线性插值成固定周期 ServoJ 流。

    Linearly resamples embedded waypoints into a fixed-period ServoJ stream.
    Times start at 0 and the exact final point is included.
    """
    times = [w[0] for w in waypoints]
    joints = [w[1] for w in waypoints]
    if len(times) < 2:
        return [(0.0, list(joints[0]))]
    duration = times[-1]
    out = []
    sample = 0.0
    index = 0
    while sample <= duration + 1e-9:
        while index < len(times) - 2 and times[index + 1] <= sample:
            index += 1
        t0, t1 = times[index], times[index + 1]
        span = t1 - t0
        frac = 0.0 if span <= 0 else max(0.0, min(1.0, (sample - t0) / span))
        target = [j0 + (j1 - j0) * frac for j0, j1 in zip(joints[index], joints[index + 1])]
        out.append((round(sample, 4), target))
        sample += period
    out[-1] = (duration, list(joints[-1]))
    return out


def limit_velocity(stream, max_speed_deg_s):
    """把相邻目标间的关节增量限制在 max_speed*dt 内 (安全限速)。

    Clamps each per-sample joint step to max_speed*dt so no velocity spike is
    ever commanded. Lower --max-speed to run the demo more slowly.
    """
    limited = [stream[0]]
    for (t1, p1), (t2, p2) in zip(stream, stream[1:]):
        dt = t2 - t1
        if dt <= 0:
            continue
        step = [p2[i] - p1[i] for i in range(6)]
        max_step = max_speed_deg_s * dt
        scale = 1.0
        for delta in step:
            if abs(delta) > max_step:
                scale = min(scale, max_step / abs(delta))
        limited.append((t2, [p1[i] + step[i] * scale for i in range(6)]))
    return limited


def stream_servo(move, stream, stop_event, triggers=(), keepalive=None,
                 keepalive_period_s=2.0):
    """按墙钟时间表发送 ServoJ；triggers 为 (时间, 回调) 列表，到达时触发一次。

    Streams the ServoJ table against a wall-clock schedule. Each (t, callback)
    in triggers fires once when the stream time passes t (used to time hand
    actions). The final target is held ~1.5 s after the last sample.
    keepalive is called every keepalive_period_s to keep a serial bus awake.
    """
    fired = [False] * len(triggers)
    started = time.monotonic()
    previous_t = 0.0
    keepalive_every = max(1, int(round(keepalive_period_s / SERVO_PERIOD_S)))
    for index, (stream_time, target) in enumerate(stream):
        if stop_event.is_set():
            raise RuntimeError("播放被用户中断 (playback stopped by user)")
        if keepalive is not None and index % keepalive_every == 0:
            keepalive()
        for i, (tt, callback) in enumerate(triggers):
            if not fired[i] and stream_time >= tt:
                fired[i] = True
                callback(stream_time)
        remaining = (started + stream_time) - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        dt = SERVO_PERIOD_S if index == 0 else stream_time - previous_t
        move.ServoJ(*target, t=SERVO_T_S, lookahead_time=LOOKAHEAD_TIME, gain=GAIN)
        previous_t = stream_time
    settle_deadline = started + stream[-1][0] + 1.5
    while time.monotonic() < settle_deadline:
        if stop_event.is_set():
            raise RuntimeError("播放被用户中断 (playback stopped by user)")
        move.ServoJ(*stream[-1][1], t=SERVO_T_S, lookahead_time=LOOKAHEAD_TIME, gain=GAIN)
        time.sleep(SERVO_PERIOD_S)



# --------------------------------------------------------------------------
# 灵巧手联动 (dexterous-hand coordination)
# --------------------------------------------------------------------------
def run_hand_pose(name, stream_time=None):
    """执行灵巧手动作（15 电机，rad）。Executes a hand pose.

    默认打印提示并等待回车（由操作员手动操作手部）；接入真实手部时给
    HAND_DRIVER 赋值即可自动下发。By default prints a prompt and waits for
    Enter (manual hand operation); assign HAND_DRIVER to drive the hand
    automatically.
    """
    positions = HAND_POSITIONS[name]
    if HAND_DRIVER is not None:
        HAND_DRIVER(list(range(1, 16)), positions)
        print("[hand] %s: 已下发 %d 电机目标 (t=%.2fs)" % (name, len(positions), stream_time or 0.0))
        return
    print("[hand] >>> 请执行手部动作: %s（%d 电机）<<<" % (name, len(positions)))
    print("[hand]     目标位置(rad): %s" % (", ".join("%.3f" % v for v in positions)))
    try:
        input("[hand] 手部动作完成后按回车继续 ...")
    except EOFError:
        pass


def default_hand_port():
    """从 CRAFT 校准文件读实际串口; 读不到则回退 COM11。

    Reads the hand's real COM port from the CRAFT calibration file
    (port_at_capture), falling back to COM11.
    """
    try:
        from pathlib import Path
        calib = (Path(__file__).resolve().parent.parent.parent
                 / "CRAFT-Hand_API" / "python"
                 / "thumb_opposition_hardware_poses.json")
        with open(calib, encoding="utf-8") as fh:
            port = json.load(fh).get("port_at_capture")
        if port:
            return port
    except Exception:
        pass
    return "COM11"


def main() -> int:
    parser = argparse.ArgumentParser(description="CR3 pick-and-place with dexterous-hand grab/release.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP,
                        help="CR3 控制器 IP (default: %s)" % DEFAULT_ROBOT_IP)
    parser.add_argument("--max-speed", type=float, default=MAX_JOINT_SPEED_DEG_S,
                        help="ServoJ 关节角速度上限 deg/s (default: %s)"
                        % MAX_JOINT_SPEED_DEG_S)
    parser.add_argument("--enable-real-execution", action="store_true",
                        help="解锁实体机器人运动（默认 dry-run）/ unlock physical motion")
    parser.add_argument("--hand", choices=("auto", "manual"), default="auto",
                        help="手部模式: auto=直接驱动 CRAFT Hand 到目标位置; "
                             "manual=打印提示等回车 (default: auto)")
    parser.add_argument("--hand-port", default=default_hand_port(),
                        help="CRAFT Hand 串口 (default: 校准文件 port_at_capture=%s)"
                        % default_hand_port())
    parser.add_argument("--hand-baud", type=int, default=57600,
                        help="CRAFT Hand 波特率 (default: 57600)")
    parser.add_argument("--hand-current-ma", type=int, default=350,
                        help="CRAFT Hand 限流 mA (default: 350)")
    args = parser.parse_args()

    stream = resample_stream(WAYPOINTS_DEG)
    stream = limit_velocity(stream, args.max_speed)
    duration = stream[-1][0]
    start_deg = np.array(stream[0][1])
    end_deg = np.array(stream[-1][1])

    print("=" * 72)
    print("PICK & PLACE")
    print("Source trajectory : trajectories/pickandplace.json")
    print("Embedded waypoints: %d   stream: %d ServoJ samples @ %.0f Hz"
          % (len(WAYPOINTS_DEG), len(stream), 1.0 / SERVO_PERIOD_S))
    print("Duration          : %.2f s (+ %d s ramp to start)" % (duration, RAMP_DURATION_S))
    print("Start joints (deg):", np.round(start_deg, 2).tolist())
    print("End joints   (deg):", np.round(end_deg, 2).tolist())
    print("=" * 72)
    if not args.enable_real_execution:
        print("[dry-run] 已生成 %d 条 ServoJ 命令，未连接机器人。" % len(stream))
        print("[dry-run] 确认工作区安全后，加 --enable-real-execution --robot-ip <IP> 执行。")
        return 0

    try:
        answer = input("将控制实体 CR3 完成【抓取-放置】，确认工作区安全后输入 YES 继续: ")
    except EOFError:
        answer = ""
    if answer.strip().upper() not in ("YES", "Y"):
        print("已取消；未连接机器人。")
        return 0

    dashboard, move = connect(args.robot_ip)
    time.sleep(0.5)
    state = read_current_joints(args.robot_ip)
    require_motion_ready(state)
    ensure_queue_running(dashboard, state)

    hand = None
    if args.hand == "auto":
        try:
            hand = CraftHandDriver(args.hand_port, args.hand_baud,
                                   args.hand_current_ma)
        except Exception as exc:
            print("[hand] 手部连接失败:", exc)
            print("[hand] 可改用 --hand manual 手动执行手部动作后继续。")
            try:
                dashboard.DisableRobot()
            except Exception:
                pass
            return 2

        def _hand_pose(name, stream_time):
            hand.write_pose(list(range(1, 16)), HAND_POSITIONS[name])
            print("[hand] %s: 已驱动手部到 %d 电机目标 (t=%.2fs)"
                  % (name, len(HAND_POSITIONS[name]), stream_time or 0.0))

        # 轨迹开始前先张开手, 准备抓取
        _hand_pose("open", 0.0)
        grab_cb = lambda st: _hand_pose("grab", st)
        open_cb = lambda st: _hand_pose("open", st)
    else:
        print("[hand] 轨迹开始前请先执行手部【四指张开】...")
        run_hand_pose("open")
        grab_cb = lambda st: run_hand_pose("grab", st)
        open_cb = lambda st: run_hand_pose("open", st)

    stop_event = threading.Event()
    try:
        real_start = state["joints_deg"].copy()
        expected_start = start_deg.copy()
        ramp_steps = max(1, int(RAMP_DURATION_S / SERVO_PERIOD_S))
        ramp = [
            (float(index * SERVO_PERIOD_S),
             (real_start + (expected_start - real_start) * index / ramp_steps).tolist())
            for index in range(ramp_steps + 1)
        ]
        offset = ramp[-1][0] + SERVO_PERIOD_S
        full_stream = ramp + [(offset + t, target) for t, target in stream]

        triggers = [
            (offset + GRAB_TRIGGER_T, grab_cb),
            (offset + OPEN_TRIGGER_T, open_cb),
        ]

        print("[robot] 缓坡到起点 %d 条，然后流式播放 %d 条 ServoJ。" % (len(ramp), len(stream)))
        keepalive = hand.keepalive if hand is not None else None
        stream_servo(move, full_stream, stop_event, triggers=triggers,
                     keepalive=keepalive)
        print("[robot] 动作完成。")
    except KeyboardInterrupt:
        stop_event.set()
        print("[robot] 已请求停止。")
        return 130
    except Exception as exc:
        print("[robot] 执行失败:", exc)
        return 1
    finally:
        if hand is not None:
            hand.close()
        try:
            dashboard.DisableRobot()
            print("[robot] 已下使能。")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
