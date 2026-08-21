#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文:
演示 CR3 机械臂完成"擦桌子"动作: 沿桌面往复擦拭。
轨迹来自 trajectories/擦桌子1.json（已精简, 线性插值误差<0.5度）。

English:
Demonstrates a table-wiping routine on a CR3 arm.
Source: trajectories/擦桌子1.json (simplified, <0.5 deg interp error).

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
# 来源: trajectories/擦桌子1.json (已精简, 线性插值误差 < 0.5 度)
# --------------------------------------------------------------------------
WAYPOINTS_DEG = [
    (0.000, [67.144, -14.314, -84.498, 16.596, 82.391, -23.845]),
    (0.938, [67.650, -14.021, -84.741, 16.476, 82.358, -23.384]),
    (1.344, [69.694, -12.882, -85.687, 16.001, 82.227, -21.526]),
    (1.922, [74.000, -10.737, -87.463, 15.042, 81.975, -17.605]),
    (2.859, [79.639, -8.517, -89.310, 13.906, 81.703, -12.466]),
    (3.656, [84.318, -7.128, -90.493, 13.073, 81.523, -8.196]),
    (3.906, [86.143, -6.695, -90.873, 12.778, 81.463, -6.530]),
    (4.484, [86.191, -7.003, -91.227, 13.323, 81.465, -6.472]),
    (4.734, [86.261, -8.288, -92.595, 15.525, 81.478, -6.348]),
    (5.875, [86.501, -12.675, -96.456, 22.393, 81.519, -5.948]),
    (6.219, [86.590, -14.310, -97.633, 24.741, 81.532, -5.806]),
    (7.188, [86.871, -19.556, -100.656, 31.661, 81.575, -5.373]),
    (8.141, [87.101, -23.964, -102.431, 36.844, 81.609, -5.032]),
    (8.391, [87.181, -25.503, -102.929, 38.552, 81.621, -4.918]),
    (8.766, [88.466, -27.533, -103.763, 40.809, 81.695, -3.681]),
    (9.078, [91.093, -29.434, -104.722, 42.899, 81.854, -1.217]),
    (9.594, [91.261, -32.222, -105.178, 45.620, 81.878, -0.998]),
    (9.953, [93.090, -32.179, -105.442, 45.593, 82.015, 0.678]),
    (10.156, [94.985, -32.094, -105.668, 45.494, 82.162, 2.413]),
    (10.609, [95.718, -32.642, -105.798, 45.985, 82.224, 3.096]),
    (11.219, [95.879, -36.121, -106.098, 49.161, 82.253, 3.315]),
    (11.422, [95.962, -37.931, -106.173, 50.744, 82.268, 3.427]),
    (11.734, [96.047, -39.814, -106.183, 52.333, 82.284, 3.541]),
    (13.141, [96.048, -39.845, -106.183, 52.359, 82.284, 3.542]),
    (13.453, [94.201, -39.872, -106.035, 52.458, 82.097, 1.854]),
    (13.656, [92.293, -39.921, -105.835, 52.545, 81.908, 0.110]),
    (13.813, [91.661, -39.947, -105.751, 52.568, 81.847, -0.469]),
    (13.953, [90.401, -40.001, -105.575, 52.611, 81.727, -1.622]),
    (14.781, [84.796, -40.412, -104.444, 52.695, 81.229, -6.764]),
    (15.297, [81.741, -40.755, -103.584, 52.662, 80.985, -9.574]),
    (15.438, [80.549, -40.911, -103.205, 52.636, 80.896, -10.672]),
    (16.281, [74.688, -41.871, -100.953, 52.378, 80.506, -16.081]),
    (17.031, [74.061, -41.997, -100.667, 52.335, 80.471, -16.660]),
    (17.750, [69.085, -43.117, -98.152, 51.916, 80.221, -21.269]),
    (18.453, [68.521, -43.263, -97.829, 51.856, 80.198, -21.792]),
    (18.844, [68.424, -41.433, -97.834, 50.327, 80.187, -21.925]),
    (18.906, [68.357, -40.174, -97.822, 49.264, 80.180, -22.017]),
    (19.094, [68.291, -38.929, -97.785, 48.190, 80.173, -22.108]),
    (19.313, [68.194, -37.096, -97.667, 46.556, 80.162, -22.245]),
    (19.922, [68.003, -33.495, -97.260, 43.199, 80.141, -22.517]),
    (20.453, [68.518, -33.275, -97.561, 43.182, 80.148, -22.040]),
    (21.125, [72.773, -31.810, -99.913, 43.234, 80.228, -18.102]),
    (21.953, [77.319, -30.542, -101.989, 43.202, 80.373, -13.901]),
    (22.313, [79.604, -30.004, -102.888, 43.159, 80.467, -11.791]),
    (22.625, [80.251, -29.862, -103.128, 43.144, 80.495, -11.194]),
    (22.984, [80.315, -30.927, -103.323, 44.197, 80.504, -11.106]),
    (23.750, [80.643, -36.729, -104.039, 49.650, 80.551, -10.653]),
    (24.234, [80.813, -39.809, -104.162, 52.328, 80.577, -10.422]),
    (25.625, [80.814, -39.840, -104.162, 52.354, 80.577, -10.419]),
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
    parser = argparse.ArgumentParser(description="CR3 table-wiping routine.")
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
    print("WIPE TABLE")
    print("Source trajectory : trajectories/擦桌子1.json")
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
        answer = input("将控制实体 CR3 完成【擦桌子】，确认工作区安全后输入 YES 继续: ")
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
