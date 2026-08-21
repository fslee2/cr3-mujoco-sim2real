#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文:
演示 CR3 机械臂完成"拧螺丝/驱动"动作: 带动工具沿目标路径连续驱动。
轨迹来自 trajectories/driver2.json（已精简, 线性插值误差<0.5度）。

English:
Demonstrates a screw-driving routine on a CR3 arm.
Source: trajectories/driver2.json (simplified, <0.5 deg interp error).

安全 / Safety:
默认 dry-run。真正运动需 --enable-real-execution 并输入 YES。程序会读取反馈端口
确认机器人使能/无报警/可运动，从当前位姿缓坡到起点，33 Hz ServoJ 限速流式播放。
请确保工作区安全。"""

from __future__ import annotations

import argparse
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

MAX_JOINT_SPEED_DEG_S = 30.0     # 不低于录制峰值速度, 避免失真


# --------------------------------------------------------------------------
# 内嵌轨迹点: (时间 s, [J1..J6] 度)
# Embedded waypoints: (time s, [J1..J6] deg)
# 来源: trajectories/driver2.json (已精简, 线性插值误差 < 0.5 度)
# --------------------------------------------------------------------------
WAYPOINTS_DEG = [
    (0.000, [99.975, -48.635, 34.214, -91.314, 93.515, 0.198]),
    (1.000, [99.364, -48.620, 34.205, -91.352, 93.334, -0.337]),
    (2.469, [87.477, -49.204, 33.935, -90.825, 89.560, -11.317]),
    (2.610, [86.721, -49.211, 33.933, -90.816, 89.544, -11.393]),
    (3.797, [86.659, -49.693, 34.145, -90.619, 89.529, -11.447]),
    (4.579, [86.037, -56.197, 36.928, -87.858, 89.397, -11.982]),
    (4.657, [86.037, -56.197, 36.928, -87.858, 89.397, -11.982]),
    (4.719, [85.943, -57.265, 37.361, -87.376, 89.376, -12.061]),
    (5.250, [85.587, -61.122, 39.068, -85.331, 89.297, -12.365]),
    (5.579, [85.366, -64.362, 40.160, -83.909, 89.248, -12.551]),
    (6.188, [85.362, -64.588, 40.184, -83.877, 89.247, -12.555]),
    (6.485, [84.825, -63.625, 40.718, -85.401, 89.104, -13.018]),
    (6.688, [84.806, -63.592, 40.736, -85.454, 89.099, -13.034]),
    (6.891, [85.410, -63.508, 40.778, -85.590, 89.299, -12.513]),
    (7.219, [87.872, -63.197, 40.933, -86.087, 90.115, -10.388]),
    (7.641, [87.903, -63.194, 40.935, -86.093, 90.125, -10.361]),
    (7.891, [87.859, -63.762, 41.151, -85.816, 90.115, -10.399]),
    (8.485, [87.471, -69.066, 43.107, -83.158, 90.018, -10.735]),
    (9.672, [87.469, -69.097, 43.117, -83.143, 90.018, -10.736]),
    (9.875, [87.508, -68.497, 42.910, -83.459, 90.028, -10.702]),
    (10.438, [87.795, -64.664, 41.429, -85.621, 90.100, -10.455]),
    (11.407, [88.481, -56.024, 38.112, -89.714, 90.272, -9.861]),
    (11.547, [88.530, -55.227, 37.885, -89.953, 90.284, -9.818]),
    (12.000, [88.536, -55.171, 37.862, -89.978, 90.286, -9.814]),
    (12.282, [87.914, -55.241, 37.836, -89.884, 90.088, -10.353]),
    (12.907, [82.514, -56.100, 37.552, -88.691, 88.128, -15.684]),
    (13.266, [79.814, -56.557, 37.412, -88.026, 87.353, -17.791]),
    (13.329, [78.734, -56.810, 37.336, -87.653, 86.970, -18.831]),
    (13.485, [78.098, -56.822, 37.333, -87.636, 86.953, -18.877]),
    (14.172, [78.667, -56.685, 37.374, -87.840, 87.135, -18.382]),
    (14.579, [81.013, -56.145, 37.536, -88.641, 87.885, -16.343]),
    (15.157, [81.041, -56.139, 37.538, -88.650, 87.894, -16.318]),
    (15.875, [81.934, -57.662, 36.964, -86.560, 88.128, -15.543]),
    (16.235, [81.956, -57.701, 36.949, -86.506, 88.134, -15.524]),
    (17.219, [81.552, -63.226, 39.042, -83.827, 88.053, -15.851]),
    (17.922, [81.211, -68.558, 40.912, -81.066, 87.983, -16.124]),
    (19.094, [81.812, -68.499, 40.976, -81.214, 88.186, -15.606]),
    (19.610, [83.013, -68.331, 41.078, -81.522, 88.590, -14.571]),
    (20.500, [83.026, -68.330, 41.079, -81.525, 88.594, -14.561]),
    (21.438, [82.882, -70.767, 41.878, -80.197, 88.562, -14.677]),
    (21.829, [82.746, -73.165, 42.650, -78.840, 88.531, -14.787]),
    (22.375, [82.743, -73.266, 42.669, -78.803, 88.530, -14.790]),
    (22.844, [82.839, -71.479, 42.109, -79.838, 88.552, -14.712]),
    (23.282, [83.015, -68.307, 41.114, -81.585, 88.592, -14.571]),
    (24.516, [83.769, -57.714, 37.171, -87.257, 88.761, -13.952]),
    (25.719, [84.541, -48.388, 33.518, -91.182, 88.923, -13.305]),
    (26.766, [84.546, -48.334, 33.494, -91.204, 88.924, -13.300]),
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


def stream_servo(move, stream, stop_event, triggers=()):
    """按墙钟时间表发送 ServoJ；triggers 为 (时间, 回调) 列表，到达时触发一次。

    Streams the ServoJ table against a wall-clock schedule. Each (t, callback)
    in triggers fires once when the stream time passes t (used to time hand
    actions). The final target is held ~1.5 s after the last sample.
    """
    fired = [False] * len(triggers)
    started = time.monotonic()
    previous_t = 0.0
    for index, (stream_time, target) in enumerate(stream):
        if stop_event.is_set():
            raise RuntimeError("播放被用户中断 (playback stopped by user)")
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




def main() -> int:
    parser = argparse.ArgumentParser(description="CR3 screw-driving routine.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP,
                        help="CR3 控制器 IP (default: %s)" % DEFAULT_ROBOT_IP)
    parser.add_argument("--max-speed", type=float, default=MAX_JOINT_SPEED_DEG_S,
                        help="ServoJ 关节角速度上限 deg/s (default: %s)"
                        % MAX_JOINT_SPEED_DEG_S)
    parser.add_argument("--enable-real-execution", action="store_true",
                        help="解锁实体机器人运动（默认 dry-run）/ unlock physical motion")
    args = parser.parse_args()

    stream = resample_stream(WAYPOINTS_DEG)
    stream = limit_velocity(stream, args.max_speed)
    duration = stream[-1][0]
    start_deg = np.array(stream[0][1])
    end_deg = np.array(stream[-1][1])

    print("=" * 72)
    print("SCREW DRIVE")
    print("Source trajectory : trajectories/driver2.json")
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
        answer = input("将控制实体 CR3 完成【拧螺丝/驱动】，确认工作区安全后输入 YES 继续: ")
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

        triggers = []

        print("[robot] 缓坡到起点 %d 条，然后流式播放 %d 条 ServoJ。" % (len(ramp), len(stream)))
        stream_servo(move, full_stream, stop_event, triggers=triggers)
        print("[robot] 动作完成。")
    except KeyboardInterrupt:
        stop_event.set()
        print("[robot] 已请求停止。")
        return 130
    except Exception as exc:
        print("[robot] 执行失败:", exc)
        return 1
    finally:
        try:
            dashboard.DisableRobot()
            print("[robot] 已下使能。")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
