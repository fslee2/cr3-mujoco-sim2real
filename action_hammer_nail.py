#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文:
演示 CR3 机械臂完成"敲钉子"动作: 持续敲击。
轨迹来自 trajectories/敲钉子2.json（已精简, 线性插值误差<0.5度）。

English:
Demonstrates a hammering routine on a CR3 arm.
Source: trajectories/敲钉子2.json (simplified, <0.5 deg interp error).

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
# 来源: trajectories/敲钉子2.json (已精简, 线性插值误差 < 0.5 度)
# --------------------------------------------------------------------------
WAYPOINTS_DEG = [
    (0.000, [80.866, -12.329, -54.902, -18.927, 93.606, -0.035]),
    (1.140, [80.648, -11.757, -55.156, -19.267, 93.619, -0.234]),
    (1.281, [80.188, -10.559, -55.684, -19.983, 93.645, -0.653]),
    (1.765, [80.407, -11.121, -55.442, -19.643, 93.633, -0.453]),
    (2.500, [82.425, -15.981, -53.064, -16.480, 93.511, 1.385]),
    (3.187, [84.638, -21.381, -50.018, -12.959, 93.335, 3.403]),
    (3.500, [84.868, -23.478, -49.674, -12.517, 93.314, 3.612]),
    (3.922, [84.772, -24.190, -50.148, -11.545, 93.321, 3.514]),
    (4.250, [84.568, -25.736, -51.149, -9.452, 93.334, 3.306]),
    (5.281, [83.877, -32.034, -54.567, -1.977, 93.367, 2.592]),
    (6.265, [83.371, -38.331, -57.045, 5.043, 93.374, 2.057]),
    (6.875, [83.107, -42.612, -58.282, 9.903, 93.370, 1.773]),
    (7.140, [82.923, -42.282, -58.652, 10.638, 93.372, 1.603]),
    (8.468, [82.492, -41.441, -59.501, 10.518, 93.376, 1.204]),
    (8.703, [82.463, -41.980, -59.635, 11.090, 93.376, 1.172]),
    (8.843, [82.405, -43.073, -59.896, 12.243, 93.375, 1.110]),
    (10.062, [81.767, -41.929, -61.139, 12.149, 93.375, 0.520]),
    (10.640, [81.686, -43.557, -61.506, 13.848, 93.373, 0.431]),
    (11.156, [81.531, -47.041, -62.176, 17.396, 93.367, 0.258]),
    (12.297, [81.530, -47.071, -62.181, 17.425, 93.367, 0.256]),
    (12.828, [81.597, -45.359, -61.911, 15.729, 93.371, 0.333]),
    (13.906, [82.015, -37.568, -60.009, 7.908, 93.381, 0.794]),
    (14.422, [82.253, -34.328, -58.826, 4.668, 93.380, 1.050]),
    (14.703, [82.440, -31.628, -57.874, 1.968, 93.375, 1.248]),
    (15.297, [82.771, -27.308, -56.153, -2.352, 93.359, 1.595]),
    (15.500, [83.068, -26.846, -55.331, -3.972, 93.340, 1.879]),
    (15.734, [83.672, -28.373, -54.321, -4.767, 93.294, 2.432]),
    (16.218, [85.024, -31.986, -51.962, -3.224, 93.211, 3.672]),
    (16.609, [84.687, -31.063, -52.587, -3.602, 93.232, 3.362]),
    (17.062, [83.391, -28.003, -54.815, -5.059, 93.307, 2.173]),
    (18.125, [80.177, -20.065, -59.999, -8.656, 93.434, -0.775]),
    (18.359, [79.257, -18.445, -61.204, -9.667, 93.454, -1.617]),
    (18.422, [78.978, -17.365, -61.676, -9.973, 93.459, -1.873]),
    (18.687, [78.480, -15.745, -62.334, -10.514, 93.464, -2.328]),
    (19.156, [78.718, -16.093, -62.036, -10.249, 93.459, -2.111]),
    (19.218, [79.202, -17.156, -61.421, -9.711, 93.449, -1.668]),
    (20.000, [82.006, -23.096, -57.728, -6.213, 93.357, 0.893]),
    (20.265, [82.201, -24.716, -57.621, -5.195, 93.348, 1.065]),
    (21.265, [81.554, -31.721, -61.141, 2.361, 93.363, 0.390]),
    (22.250, [81.026, -38.918, -63.909, 9.921, 93.349, -0.174]),
    (22.687, [81.138, -41.310, -63.981, 13.161, 93.353, -0.086]),
    (23.593, [81.306, -43.439, -63.878, 15.525, 93.356, 0.054]),
    (24.359, [81.231, -45.173, -64.206, 17.287, 93.353, -0.030]),
    (25.109, [81.253, -44.603, -64.111, 16.718, 93.354, -0.004]),
    (25.375, [81.324, -42.987, -63.808, 15.099, 93.357, 0.075]),
    (25.718, [81.483, -39.747, -63.084, 11.859, 93.364, 0.252]),
    (26.203, [81.633, -36.776, -62.352, 8.555, 93.367, 0.415]),
    (26.828, [81.768, -34.655, -61.660, 6.162, 93.369, 0.561]),
    (26.953, [81.769, -34.648, -61.658, 6.155, 93.369, 0.562]),
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
    parser = argparse.ArgumentParser(description="CR3 hammering routine.")
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
    print("HAMMER NAIL")
    print("Source trajectory : trajectories/敲钉子2.json")
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
        answer = input("将控制实体 CR3 完成【敲钉子】，确认工作区安全后输入 YES 继续: ")
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
