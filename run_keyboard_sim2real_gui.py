"""Custom Tk GUI for CR3 MuJoCo simulation, recording, and guarded hardware control."""

from __future__ import annotations

import argparse
import ipaddress
import math
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import mujoco
import numpy as np
from PIL import Image, ImageTk

import run_keyboard_sim2real as core
from dobot_api import alarmAlarmJsonFile
from cr3_sim2real.hardware import (
    CR3_MAX_JOINT_SPEED_DEG_S,
    DEFAULT_LIVE_JOINT_SPEED_DEG_S,
    FeedbackReceiver,
    LIVE_SERVO_PERIOD_S,
    LiveServoHardware,
    clear_robot_error,
    continue_robot_queue,
    disable_robot,
    emergency_stop_robot,
    enable_robot,
    get_robot_error_ids,
    joint_target_reached,
    pause_robot_queue,
    power_on_robot,
    require_motion_ready,
    set_robot_speed_factor,
    start_drag_mode,
    stop_drag_mode,
    stream_live_target_until_reached,
    wait_for_drag_state,
)
from cr3_sim2real.hamer_bridge import (
    HamerArmMapper,
    HamerBridgeTracker,
    hamer_cam_t,
)
from cr3_sim2real.joint_mapping import (
    MAPPING_CALIBRATED,
    real_deg_to_sim_rad,
    sim_rad_to_real_deg,
)
from cr3_sim2real.quest_hand import QuestHandReceiver, QuestWristMapper
from cr3_sim2real.trajectory import (
    TrajectoryRecorder,
    build_servoj_dry_run,
    validate_trajectory,
)


VIEW_WIDTH = 960
VIEW_HEIGHT = 720
MIN_RENDER_WIDTH = 320
MIN_RENDER_HEIGHT = 240
UI_PERIOD_MS = 10
RENDER_PERIOD_S = 1.0 / 30.0
RENDER_RESIZE_DEBOUNCE_MS = 140
KEY_HOLD_DELAY_S = 0.18
KEY_REPEAT_PERIOD_S = 0.05
QUEST_CONTROL_RATE_HZ = 60.0
QUEST_DEFAULT_CARTESIAN_SPEED_M_S = 0.80
QUEST_MAX_CONTROL_DT_S = 0.05
# Keep the default Quest path position-based and deterministic.  The
# controller has no need to extrapolate a hand pose for the current UDP link;
# a single fixed EMA is easier to tune than speed-dependent filtering.
QUEST_DEFAULT_FILTER_ALPHA = 0.75
QUEST_MAX_PREDICT_S = 0.0
HAMER_REAL_HOME_SPEED_DEG_S = 5.0
MANUAL_REAL_HOME_SPEED_DEG_S = 5.0
HAMER_REAL_HAND_WATCHDOG_S = 5.0
HAMER_REAL_CONFIRMATION_WINDOW_S = 30.0
# The two CR3 controllers on the project's robot LAN.  Keep these as
# selectable defaults while still allowing a manually entered IP.
KNOWN_ROBOT_IPS = ("192.168.5.11", "192.168.5.12")
DEFAULT_HAMER_ORIGIN_VIDEO = (
    core.ROOT.parent.parent / "mujoco_ws" / "hand_sequence.avi"
)

ROBOT_MODE_NAMES = {
    1: "INIT",
    2: "BRAKE_OPEN",
    3: "POWER_OFF",
    4: "DISABLED",
    5: "ENABLED_IDLE",
    6: "BACKDRIVE",
    7: "RUNNING",
    8: "RECORDING",
    9: "ERROR",
    10: "PAUSE",
    11: "JOG",
}


UI_TEXT_EN = {
    "键盘待命：W/S=Z  A/D=Y  Q/E=X  I/K=RX  J/L=RY  U/O=RZ":
        "Keyboard ready: W/S=Z  A/D=Y  Q/E=X  I/K=RX  J/L=RY  U/O=RZ",
    "键盘控制已恢复：W/S=Z  A/D=Y  Q/E=X  I/K=RX  J/L=RY  U/O=RZ":
        "Keyboard control restored: W/S=Z  A/D=Y  Q/E=X  I/K=RX  J/L=RY  U/O=RZ",
    "正在编辑参数；Esc=强制取消使能":
        "Editing parameters; Esc = immediate Disable",
    "窗口失焦：运动键已清除": "Window unfocused: motion keys cleared",
    "实机反馈未连接": "Robot feedback disconnected",
    "J1—J6：等待实机反馈": "J1—J6: waiting for robot feedback",
    "TCP：等待实机反馈": "TCP: waiting for robot feedback",
    "控制器遥测：等待实机反馈": "Controller telemetry: waiting for feedback",
    "关节力学：等待实机反馈": "Joint mechanics: waiting for feedback",
    "TCP / 六维力：等待实机反馈": "TCP / 6-axis force: waiting for feedback",
    "未武装": "Not armed",
    "实时同步未启动": "Live sync inactive",
    "示教拖拽未启动": "Teaching inactive",
    "实机连接与安全": "Robot Connection & Safety",
    "连接 / 切换": "Connect / Switch",
    "断开反馈": "Disconnect",
    "仿真对齐实机": "Align Sim",
    "机器人上电": "Power On",
    "使能": "Enable",
    "① 机器人上电": "1  Power On",
    "② 使能": "2  Enable",
    "取消使能  [Esc]": "Disable  [Esc]",
    "读取报警": "Alarms",
    "清除报警": "Clear",
    "暂停队列": "Pause",
    "继续队列": "Continue",
    "实机回 Home": "Robot Home",
    "停止实机回 Home": "Stop Robot Home",
    "正在停止实机 Home…": "Stopping Robot Home…",
    "目标 IP（可输入或选择）": "Target IP (type or select)",
    "未连接 · 输入 IP 后点击“连接 / 切换”":
        "Disconnected · enter an IP, then click Connect / Switch",
    "正在连接…": "Connecting…",
    "推荐顺序：① 输入/选择 IP  ② 连接/切换  ③ 上电  ④ 使能  ⑤ 对齐或选择控制模式\n切换机械臂：先停止运动并建议取消使能，再输入新 IP 并点击“连接 / 切换”":
        "Recommended: 1 enter/select IP  2 connect/switch  3 power on  4 enable  5 align or choose a mode\nTo switch robots: stop motion and preferably disable it, then enter the new IP and click Connect / Switch",
    "控制模式": "Control Mode",
    "录制后回放": "Record",
    "实时同步": "Live",
    "示教拖拽": "Teach",
    "HaMeR 仿真": "HaMeR",
    "HaMeR 参数…": "Settings…",
    "HaMeR 实时仿真": "HaMeR Camera",
    "停止 HaMeR 实时": "Stop HaMeR Live",
    "正在打开摄像头…": "Opening Camera…",
    "Testing HaMeR": "Testing HaMeR",
    "正在打开测试视频…": "Opening Test Video…",
    "原点已确定 · 验证中": "Origin Set · Validating",
    "停止测试视频": "Stop Test",
    "HaMeR 实机同步": "HaMeR Robot Sync",
    "停止 HaMeR 实机同步": "Stop HaMeR Robot Sync",
    "停止回 Home": "Stop Homing",
    "等待手部原点…": "Waiting for Hand Origin…",
    "最终确认 HaMeR 实机同步": "Final Confirm HaMeR Robot Sync",
    "Meta Quest 仿真": "Meta Quest",
    "Quest 设置…": "Quest Settings…",
    "启动 Quest 接收": "Start Quest Receiver",
    "停止 Quest 接收": "Stop Quest Receiver",
    "设定腕部原点  [R]": "Set Wrist Origin  [R]",
    "Quest 实机同步": "Quest Robot Sync",
    "最终确认 Quest 实机同步": "Final Confirm Quest Robot Sync",
    "Quest 实机限速 (deg/s)": "Quest Speed Limit (deg/s)",
    "启动实时同步": "Start Live",
    "停止实时同步": "Stop Live",
    "取消连接": "Cancel Connection",
    "进入示教拖拽": "Enter Teach",
    "退出示教拖拽": "Exit Teach",
    "重试退出示教": "Retry Exit",
    "参数调节": "Parameters",
    "MuJoCo 速度倍率": "Sim Speed",
    "平移步长 (mm)": "Move Step",
    "旋转步长 (deg)": "Rotate Step",
    "回放 SpeedJ (%)": "SpeedJ",
    "回放 AccJ (%)": "AccJ",
    "实时限速 (deg/s)": "Live Limit",
    "实机全局倍率 (%)": "Global Scale",
    "MuJoCo 倍速": "Sim Speed",
    "平移 (mm)": "Move Step",
    "旋转 (deg)": "Rotate Step",
    "回放 SpeedJ": "SpeedJ",
    "回放 AccJ（保留）": "AccJ (Reserved)",
    "同步限速": "Live Limit",
    "全局倍率": "Global Scale",
    "应用实机全局倍率": "Apply Global Scale",
    "仿真与轨迹": "Simulation & Trajectory",
    "开始记录  [KP Enter]": "Record  [Enter]",
    "停止记录  [KP Enter]": "Stop  [Enter]",
    "仿真 Home  [KP 5]": "Home  [KP 5]",
    "清空轨迹  [KP /]": "Clear  [KP /]",
    "武装真实回放": "Arm Playback",
    "最终确认执行": "Execute",
    "重置相机": "Reset Camera",
    "运行日志": "Runtime Log",
    "清空日志": "Clear Log",
    "软件紧急停止  EMERGENCY STOP": "SOFTWARE EMERGENCY STOP",
    "鼠标：左键旋转  ·  右键平移  ·  滚轮缩放":
        "Mouse: left rotate · right pan · wheel zoom",
    "笛卡尔点动 · 按住连续移动，松开停止":
        "Cartesian Jog · hold to move, release to stop",
    "软件急停不能替代实体急停按钮；解除急停/清报警请在 DobotStudio 手动完成。":
        "Software stop does not replace the physical E-stop. Reset E-stop and alarms in DobotStudio.",
    "30004 连接失败": "30004 connection failed",
    "J1—J6：反馈连接失败": "J1—J6: feedback connection failed",
    "TCP：反馈连接失败": "TCP: feedback connection failed",
    "控制器遥测：反馈连接失败": "Controller telemetry: connection failed",
    "关节力学：反馈连接失败": "Joint mechanics: connection failed",
    "TCP / 六维力：反馈连接失败": "TCP / 6-axis force: connection failed",
    "Testing HaMeR=测试视频 · 实时摄像头中按 R 设定原点":
        "Testing HaMeR = test video · press R in live camera to set origin",
    "Quest 未启动 · 仅控制 MuJoCo":
        "Quest inactive · MuJoCo only",
}
UI_TEXT_ZH = {english: chinese for chinese, english in UI_TEXT_EN.items()}


# Runtime logs are translated at the single ``log`` entry point so messages
# from background callbacks follow the currently selected GUI language too.
LOG_TEXT_EN = {
    "自定义 MuJoCo GUI 已启动；当前没有发送实机命令。":
        "Custom MuJoCo GUI started; no real-robot command has been sent.",
    "实机运动锁定：启动时需要 --enable-real-execution。":
        "Real motion is locked; start with --enable-real-execution.",
    "控制已锁定：真实回放中或软件急停已锁存。":
        "Control is locked by real playback or the latched software E-stop.",
    "请先点击“启动实时同步”。": "Click Start Live Sync first.",
    "示教拖拽模式禁用键盘和 GUI 点动，请直接安全地拖动实体机械臂。":
        "Keyboard and GUI jogging are disabled in teaching mode; move the physical robot by hand.",
    "HaMeR 模式由手部位移独占控制；请停止 HaMeR 实时仿真或切换模式后再键盘点动。":
        "HaMeR exclusively controls motion; stop HaMeR live simulation or change mode before jogging.",
    "实时同步或示教拖拽模式禁用 Home。": "Home is disabled during live sync or teaching.",
    "HaMeR 实机接管流程中禁用手动 Home；请使用 HaMeR 实机同步按钮停止。":
        "Manual Home is disabled during HaMeR robot handover; use the HaMeR Robot Sync button to stop.",
    "MuJoCo 正在返回 Home；实体机器人不受影响。":
        "MuJoCo is returning Home; the physical robot is unaffected.",
    "实机回 Home 已锁定：请使用 --enable-real-execution 启动 GUI。":
        "Robot Home is locked; start the GUI with --enable-real-execution.",
    "当前存在其他实体运动流程；停止后才能让实机回 Home。":
        "Another physical-robot motion workflow is active; stop it before Robot Home.",
    "关节映射未标定，禁止实体 CR3 回 Home。":
        "Joint mapping is not calibrated; physical CR3 Home is prohibited.",
    "正在先连接 30004 只读反馈；连接后请再次点击“实机回 Home”。":
        "Connecting port 30004 feedback first; click Robot Home again afterward.",
    "用户取消了实机回 Home。": "The user cancelled Robot Home.",
    "正在停止实机回 Home；不会继续发送新的 ServoJ 目标。":
        "Stopping Robot Home; no new ServoJ targets will be sent.",
    "实体 CR3 已到 Home；MuJoCo 已对齐，已有手部原点已重新锚定到 Link6。":
        "The physical CR3 reached Home; MuJoCo is aligned and existing hand origins were re-anchored to Link6.",
    "实机回 Home 已由用户停止。": "Robot Home was stopped by the user.",
    "DisableRobot() 已在发送中。": "DisableRobot() is already being sent.",
    "只有仿真录制模式，或已验证的示教拖拽模式可以记录轨迹。":
        "Trajectories can be recorded only in simulation record mode or verified teaching mode.",
    "示教轨迹记录已开始。": "Teaching trajectory recording started.",
    "轨迹记录已开始。": "Trajectory recording started.",
    "轨迹已清空；实体机器人未收到命令。":
        "Trajectory cleared; the physical robot received no command.",
    "没有已停止并可供检查的轨迹。": "No stopped trajectory is available for review.",
    "轨迹安全检查失败：": "Trajectory safety check failed:",
    "尚未连接 30003，尚未发送运动命令。":
        "Port 30003 is not connected; no motion command has been sent.",
    "实机运动锁定：请使用 --enable-real-execution 启动 GUI。":
        "Real motion is locked; start the GUI with --enable-real-execution.",
    "软件急停已锁存；请在 DobotStudio 清除并重新使能。":
        "Software E-stop is latched; reset it and re-enable the robot in DobotStudio.",
    "请先退出示教拖拽，再武装真实回放。": "Exit teaching before arming real playback.",
    "HaMeR 当前是仿真专用模式；请停止 HaMeR 实时仿真并切换到录制回放后再武装实机。":
        "HaMeR is currently in simulation mode; stop it and switch to record/playback before arming playback.",
    "请先完成轨迹记录和 Dry Run。": "Complete trajectory recording and Dry Run first.",
    "关节映射未标定，禁止真实回放。": "Joint mapping is not calibrated; real playback is prohibited.",
    "真实回放已武装 15 秒；尚未发送运动命令。":
        "Real playback is armed for 15 seconds; no motion command has been sent.",
    "真实回放武装已过期，请重新武装。": "Real playback arming expired; arm it again.",
    "用户取消了最终真实回放确认。": "The user cancelled final real-playback confirmation.",
    "最终确认完成，正在连接并执行低速真实回放。":
        "Final confirmation accepted; connecting and starting low-speed real playback.",
    "真实回放已完成。": "Real playback completed.",
    "30004 只读反馈正常，无需重复连接。": "Port 30004 feedback is healthy; no reconnection is needed.",
    "30004 只读反馈已连接。": "Port 30004 read-only feedback connected.",
    "请先输入目标 IP，并点击“连接 / 切换”。":
        "Enter the target IP and click Connect / Switch first.",
    "当前没有已连接的机器人。": "No robot is currently connected.",
    "目标 IP 已修改；真实回放武装已取消。":
        "The target IP changed; real-playback arming was cancelled.",
    "内部连接状态不一致；为安全起见已阻止实机命令，请重新连接反馈。":
        "The internal connection state is inconsistent; robot commands were blocked for safety. Reconnect feedback.",
    "内部连接状态不一致；请使用实体停止并重新连接反馈。":
        "The internal connection state is inconsistent; use the physical stop and reconnect feedback.",
    "当前有实机命令或运动流程正在运行，停止后才能切换机器人 IP。":
        "A robot command or motion workflow is active; stop it before switching robot IP.",
    "反馈连接正在建立，请等待结果后再断开。":
        "A feedback connection is being established; wait for the result before disconnecting.",
    "当前有实机命令或运动流程正在运行，停止后才能断开反馈。":
        "A robot command or motion workflow is active; stop it before disconnecting feedback.",
    "当前没有已连接的 30004 反馈。": "No port 30004 feedback is connected.",
    "没有有效的实机 IP；软件急停只完成了本地锁存，请使用实体急停。":
        "No valid robot IP is available; only the local software-stop latch was set. Use the physical E-stop.",
    "示教连接 IP 已丢失；请使用 DobotStudio 退出拖拽模式。":
        "The teaching connection IP was lost; exit drag mode in DobotStudio.",
    "示教连接 IP 已丢失；GUI 保持打开，请使用 DobotStudio 检查拖拽状态。":
        "The teaching connection IP was lost; the GUI will remain open. Check drag mode in DobotStudio.",
    "请先连接只读反馈。": "Connect read-only feedback first.",
    "记录或回放过程中不能重新对齐。": "Alignment is disabled during recording or playback.",
    "MuJoCo 已对齐到实体 CR3 当前关节姿态。":
        "MuJoCo aligned to the physical CR3 joint pose.",
    "另一个控制器命令正在执行，请稍候。": "Another controller command is running; please wait.",
    "另一个机器人状态命令正在执行，请稍候。": "Another robot-state command is running; please wait.",
    "正在通过 GetErrorID() 读取控制器和六轴伺服报警。":
        "Reading controller and six-axis servo alarms with GetErrorID().",
    "GetErrorID：当前没有活动报警。": "GetErrorID: no active alarms.",
    "清除报警已锁定：请使用 --enable-real-execution 启动 GUI。":
        "Clear Alarm is locked; start the GUI with --enable-real-execution.",
    "请先停止真实回放、实时同步或示教拖拽，再清除报警。":
        "Stop real playback, live sync, and teaching before clearing alarms.",
    "正在发送 ClearError()；不会自动 Continue 或 Enable。":
        "Sending ClearError(); Continue and Enable will not be sent automatically.",
    "暂停队列已锁定：请使用 --enable-real-execution 启动 GUI。":
        "Pause Queue is locked; start the GUI with --enable-real-execution.",
    "请先退出示教拖拽，再暂停运动队列。": "Exit teaching before pausing the motion queue.",
    "本地运动发送已停止，正在发送 pause()。":
        "Local motion transmission stopped; sending pause().",
    "继续队列已锁定：请使用 --enable-real-execution 启动 GUI。":
        "Continue Queue is locked; start the GUI with --enable-real-execution.",
    "活动运动流程会自行管理队列，此时不能手动 Continue。":
        "The active motion workflow manages the queue; manual Continue is disabled.",
    "正在发送 continue()。": "Sending Continue().",
    "实机全局倍率已锁定：请使用 --enable-real-execution 启动 GUI。":
        "Robot global scale is locked; start the GUI with --enable-real-execution.",
    "请先停止真实回放或实时同步，再修改实机全局倍率。":
        "Stop real playback or live sync before changing the robot global scale.",
    "使能已锁定：请使用 --enable-real-execution 启动 GUI。":
        "Enable is locked; start the GUI with --enable-real-execution.",
    "实体运动连接活动时不需要重复使能。": "The robot is already under an active motion connection.",
    "正在发送 EnableRobot()...": "Sending EnableRobot()...",
    "上电已锁定：请使用 --enable-real-execution 启动 GUI。":
        "Power On is locked; start the GUI with --enable-real-execution.",
    "实体运动连接活动时不能重复执行上电。": "Power On is disabled during an active motion connection.",
    "正在发送 PowerOn()；请等待末端指示灯状态稳定。":
        "Sending PowerOn(); wait for the tool indicator to stabilize.",
    "机器人已收到上电命令；状态稳定后再点击“使能 Enable”。":
        "The robot accepted PowerOn; click Enable after the state stabilizes.",
    "取消使能已锁定：请使用 --enable-real-execution 启动 GUI。":
        "Disable is locked; start the GUI with --enable-real-execution.",
    "本地运动发送已停止，正在发送 DisableRobot()。":
        "Local motion transmission stopped; sending DisableRobot().",
    "机器人已正常取消使能；无需使用软件急停。":
        "Robot disabled normally; the software E-stop is not required.",
    "软件急停已经锁存。": "Software E-stop is already latched.",
    "软件急停触发：本地运动发送已锁定，正在发送 EmergencyStop()。":
        "Software E-stop triggered; local motion is locked and EmergencyStop() is being sent.",
    "请在 DobotStudio 手动解除急停、清除报警，然后重新使能。":
        "Reset E-stop and alarms in DobotStudio, then re-enable the robot.",
    "原点设置/验证视频正在播放，请等待视频结束。":
        "The origin setup/validation video is playing; wait for it to finish.",
    "HaMeR 实机同步锁定：请使用 --enable-real-execution 启动 GUI。":
        "HaMeR Robot Sync is locked; start the GUI with --enable-real-execution.",
    "软件急停、真实回放或示教拖拽活动时不能启动 HaMeR 实机同步。":
        "HaMeR Robot Sync cannot start during E-stop, real playback, or teaching.",
    "请先启动 HaMeR 实时仿真，在摄像头中按 R，并完成仿真验证。":
        "Start HaMeR live simulation, press R in the camera view, and validate simulation first.",
    "HaMeR 摄像头未运行。": "The HaMeR camera is not running.",
    "另一个实时实机连接已经活动，请先停止。": "Another live robot connection is active; stop it first.",
    "正在先连接 30004 只读反馈；连接后请再次点击 HaMeR 实机同步。":
        "Connecting port 30004 feedback first; click HaMeR Robot Sync again afterward.",
    "用户取消了 HaMeR 实机同步第一次确认。":
        "The user cancelled the first HaMeR Robot Sync confirmation.",
    "第一次确认完成：已按 30004 自动对齐，正在以 5°/s 回统一 Home。":
        "First confirmation accepted: aligned from 30004 and returning to shared Home at 5 deg/s.",
    "用户取消了 HaMeR 实机同步最终确认。":
        "The user cancelled final HaMeR Robot Sync confirmation.",
    "HaMeR 实机同步已启动；当前控制链为 HaMeR → MuJoCo IK → CR3 ServoJ。":
        "HaMeR Robot Sync started: HaMeR → MuJoCo IK → CR3 ServoJ.",
    "原点视频正在打开，请稍候再按第二次。": "The origin video is opening; wait before the second click.",
    "真实回放或示教拖拽活动时不能设置 HaMeR 原点。":
        "The HaMeR origin cannot be set during real playback or teaching.",
    "请先停止实时实机同步；HaMeR 当前仅控制 MuJoCo。":
        "Stop live robot sync first; HaMeR currently controls MuJoCo only.",
    "请先停止 HaMeR 实时仿真，再重新设置原点。":
        "Stop HaMeR live simulation before resetting the origin.",
    "原点验证视频正在控制 MuJoCo，请等待播放完成。":
        "The origin validation video is controlling MuJoCo; wait for playback to finish.",
    "原点设置视频已在处理。": "The origin setup video is already being processed.",
    "原点视频正在打开；画面出现后，第二次点击同一按钮确定当前帧为原点。":
        "Opening the origin video; click the same button again to use the displayed frame as origin.",
    "真实回放或示教拖拽活动时不能启动 HaMeR 实时仿真。":
        "HaMeR live simulation cannot start during real playback or teaching.",
    "请先停止实时实机同步；HaMeR 实时模式当前仅控制 MuJoCo。":
        "Stop live robot sync first; HaMeR live mode currently controls MuJoCo only.",
    "HaMeR 实时摄像头已启动；按 R 之前只显示画面，不移动 MuJoCo。":
        "HaMeR live camera started; before R is pressed it displays video without moving MuJoCo.",
    "HaMeR 实时仿真已停止。": "HaMeR live simulation stopped.",
    "当前没有正在播放的 Testing HaMeR 视频。": "No Testing HaMeR video is currently playing.",
    "Testing HaMeR 视频已手动停止；MuJoCo 保持当前位姿。":
        "Testing HaMeR video stopped manually; MuJoCo holds its current pose.",
    "请先启动“HaMeR 实时仿真”，然后在摄像头画面中按 R 设定原点。":
        "Start HaMeR Live Simulation, then press R in the camera view to set the origin.",
    "请先点击“原点设置视频”打开视频。": "Click the origin video button first.",
    "原点视频将继续播放；后续手部位移现在会直接驱动 MuJoCo 机械臂。":
        "The origin video will continue; subsequent hand motion now drives the MuJoCo arm.",
    "Testing HaMeR 验证已完成；MuJoCo 保持最终位姿。":
        "Testing HaMeR validation completed; MuJoCo holds the final pose.",
    "请先点击“退出示教拖拽”，再切换控制模式。": "Exit teaching before changing control mode.",
    "示教模式命令正在执行，请稍候。": "A teaching command is running; please wait.",
    "示教拖拽已锁定：请使用 --enable-real-execution 启动 GUI。":
        "Teaching is locked; start the GUI with --enable-real-execution.",
    "软件急停、真实回放或实时同步活动时不能进入示教拖拽。":
        "Teaching cannot start during E-stop, real playback, or live sync.",
    "进入示教前请先连接 30004 只读反馈。": "Connect port 30004 feedback before teaching.",
    "正在发送 StartDrag()；等待 30004 返回拖拽状态。":
        "Sending StartDrag(); waiting for drag state on port 30004.",
    "请直接拖动实体机械臂；MuJoCo 将通过 30004 镜像当前姿态。":
        "Move the physical robot by hand; MuJoCo mirrors its pose through port 30004.",
    "退出示教已锁定：请使用 --enable-real-execution 启动 GUI。":
        "Exit Teaching is locked; start the GUI with --enable-real-execution.",
    "正在发送 StopDrag()。": "Sending StopDrag().",
    "实时同步锁定：请使用 --enable-real-execution 启动 GUI。":
        "Live sync is locked; start the GUI with --enable-real-execution.",
    "软件急停、真实回放或示教拖拽活动时不能启动实时同步。":
        "Live sync cannot start during E-stop, real playback, or teaching.",
    "当前 HaMeR 仅允许控制 MuJoCo；请先停止 HaMeR 实时仿真再启动实时实机同步。":
        "HaMeR currently controls MuJoCo only; stop it before starting live robot sync.",
    "正在先建立共享的 30004 反馈；连接完成后请再次点击“启动实时同步”。":
        "Connecting shared port 30004 feedback first; click Start Live Sync again afterward.",
    "实时同步已停止，本地不再发送 ServoJ。": "Live sync stopped; ServoJ is no longer transmitted.",
    "真实回放武装已过期。": "Real playback arming expired.",
    "正在进入示教拖拽，请等待状态验证完成后再关闭 GUI。":
        "Teaching is starting; wait for state verification before closing the GUI.",
    "示教模式命令正在执行，请稍候再退出。": "A teaching command is running; wait before exiting.",
    "用户取消了 HaMeR 实机接管流程。": "The user cancelled the HaMeR robot handover.",
    "用户停止了 HaMeR 实机同步。": "The user stopped HaMeR Robot Sync.",
    "HaMeR 实机同步第二次确认已超时。": "The second HaMeR Robot Sync confirmation timed out.",
    "HaMeR 实机同步第二次确认已超时，请重新准备。":
        "The second HaMeR Robot Sync confirmation timed out; prepare again.",
    "界面语言已切换为英文。": "Interface language switched to English.",
    "界面语言已切换为中文。": "Interface language switched to Chinese.",
    "HaMeR 映射参数必须是数字": "HaMeR mapping parameters must be numeric",
    "HaMeR 映射参数必须是正的有限数":
        "HaMeR mapping parameters must be positive and finite",
    "HaMeR EMA 系数必须在 (0, 1] 内":
        "The HaMeR EMA coefficient must be in (0, 1]",
    "原点定义视频路径不能为空":
        "The origin-definition video path cannot be empty",
    "Quest 接收已停止。": "Quest receiver stopped.",
    "Quest 实时接收已启动；按 R 前只显示腕部数据，不移动 MuJoCo。":
        "Quest receiver started; before R is pressed, wrist data is displayed without moving MuJoCo.",
    "请先启动 Quest 接收。": "Start the Quest receiver first.",
    "Quest 当前没有新鲜腕部数据；请在头显中启动 Streaming。":
        "No fresh Quest wrist data is available; start Streaming in the headset.",
    "Quest 腕部原点已清除。": "Quest wrist origin cleared.",
    "Quest 模式由腕部位移独占控制；按 R 可重设原点，或停止 Quest 接收后再点动。":
        "Quest wrist motion exclusively controls this mode; press R to reset the origin or stop the Quest receiver before jogging.",
    "请先停止 Quest 接收，再修改网络设置。":
        "Stop the Quest receiver before changing network settings.",
    "实机回放、实时同步或示教活动时不能启动 Quest 仿真。":
        "Quest simulation cannot start during real playback, live sync, or teaching.",
    "Quest 仿真正在运行；请先停止 Quest 接收并切换到录制回放模式。":
        "Quest simulation is running; stop the Quest receiver and switch to record/playback first.",
    "Quest 实机同步已启动；当前控制链为 Quest → MuJoCo IK → CR3 ServoJ。":
        "Quest Robot Sync started: Quest → MuJoCo IK → CR3 ServoJ.",
    "Quest 实机同步第二次确认已超时。":
        "The second Quest Robot Sync confirmation timed out.",
    "Quest 实机同步锁定：请使用 --enable-real-execution 启动 GUI。":
        "Quest Robot Sync is locked; start the GUI with --enable-real-execution.",
    "软件急停、真实回放或示教拖拽活动时不能启动 Quest 实机同步。":
        "Quest Robot Sync cannot start during E-stop, real playback, or teaching.",
    "另一个手部实机接管流程正在运行。":
        "Another hand-controlled robot handover is running.",
    "请先启动 Quest 接收，按 R 设定原点，并完成 MuJoCo 仿真验证。":
        "Start the Quest receiver, press R to set the origin, and validate MuJoCo first.",
    "正在先连接 30004 只读反馈；连接后请再次点击 Quest 实机同步。":
        "Connecting port 30004 feedback first; click Quest Robot Sync again afterward.",
    "用户取消了 Quest 实机同步第一次确认。":
        "The user cancelled the first Quest Robot Sync confirmation.",
    "Quest 第一次确认完成：已按 30004 自动对齐，正在以 5°/s 回统一 Home。":
        "First Quest confirmation accepted: aligned from 30004 and returning to shared Home at 5 deg/s.",
    "用户取消了 Quest 实机同步最终确认。":
        "The user cancelled final Quest Robot Sync confirmation.",
    "手部实机接管流程中禁用手动 Home；请先停止实机同步。":
        "Manual Home is disabled during a hand-controlled robot handover; stop robot synchronization first.",
}

# Long, specific fragments must appear before short generic fragments.
LOG_FRAGMENT_EN = (
    ("只读反馈已连接；后续实机命令已锁定到该 IP。", "read-only feedback connected; subsequent robot commands are locked to this IP."),
    ("目标 IP ", "Target IP "),
    (" 尚未生效；当前连接仍是 ", " has not been applied; the active connection is "),
    ("请先点击“连接 / 切换”", "Click Connect / Switch first"),
    (" 已连接，但输入 IP 已改变；该连接未启用。", " connected, but the entered IP changed; this connection was not activated."),
    ("已断开 ", "Disconnected "),
    (" 的 30004 反馈；正在切换到 ", " port 30004 feedback; switching to "),
    ("；不会再向该机器人发送命令。", "; commands will no longer be sent to that robot."),
    ("正在连接 ", "Connecting to "),
    ("，请稍候。", ", please wait."),
    ("无效的机器人 IP：", "Invalid robot IP: "),
    ("机器人 IP 必须是 IPv4 地址", "Robot IP must be an IPv4 address"),
    ("实体 CR3 和 MuJoCo 已到统一 Home；验证阶段的仿真偏移已丢弃。", "The physical CR3 and MuJoCo reached shared Home; simulation-validation offset was discarded. "),
    ("下一帧有效 HaMeR 手位将自动成为新的实机控制原点。", "The next valid HaMeR hand pose will become the new robot-control origin."),
    ("Home 不清除 HaMeR 手部原点；", "Home keeps the HaMeR hand origin; "),
    ("Link6 基准已更新为", "Link6 anchor updated to"),
    ("HaMeR 实机原点已自动重建：", "HaMeR robot origin rebuilt automatically: "),
    ("HaMeR 实时摄像头原点已设定：", "HaMeR live-camera origin set: "),
    ("HaMeR 原点已由第二次按键时的视频帧确定：", "HaMeR origin set from the video frame selected by the second click: "),
    ("当前视频帧没有有效手部 cam_t；", "The current video frame has no valid hand cam_t; "),
    ("实时摄像头当前没有有效手部 cam_t；", "The live camera has no valid hand cam_t; "),
    ("桥接状态：", "bridge status: "),
    ("最近推理耗时：", "latest inference time: "),
    ("记录完成：", "Recording completed: "),
    ("已保存：", "Saved: "),
    ("Dry Run 通过：", "Dry Run passed: "),
    ("个连续 ServoJ 采样", " continuous ServoJ samples"),
    ("个 ServoJ 采样", " ServoJ samples"),
    ("还有", "remaining"),
    ("主机限速", "host speed limit"),
    ("仅保留，ServoJ 不使用", "retained only; unused by ServoJ"),
    ("参数无效：", "Invalid parameter: "),
    ("无法将渲染分辨率调整为", "Could not resize rendering resolution to "),
    ("真实回放中止：", "Real playback aborted: "),
    ("实机回 Home 前状态检查失败：", "Robot Home precheck failed: "),
    ("实体 CR3 正在低速回 Home：", "Physical CR3 is returning Home at low speed: "),
    ("实机回 Home 已由停止命令中止：", "Robot Home was aborted by a stop command: "),
    ("实机回 Home 失败：", "Robot Home failed: "),
    ("现有 30004 反馈不可用，正在重新建立连接：", "Existing port 30004 feedback is unavailable; reconnecting: "),
    ("只读反馈连接失败：", "Read-only feedback connection failed: "),
    ("无法读取新鲜实机反馈：", "Could not read fresh robot feedback: "),
    ("GetErrorID：发现", "GetErrorID found "),
    ("条活动报警：", " active alarm(s):"),
    ("读取报警失败：", "Alarm read failed: "),
    ("正在发送 SpeedFactor(", "Sending SpeedFactor("),
    ("EnableRobot 成功：", "EnableRobot succeeded: "),
    ("EnableRobot 失败：", "EnableRobot failed: "),
    ("PowerOn 成功：", "PowerOn succeeded: "),
    ("PowerOn 失败：", "PowerOn failed: "),
    ("DisableRobot 成功：", "DisableRobot succeeded: "),
    ("DisableRobot 失败：", "DisableRobot failed: "),
    ("EmergencyStop 已被控制器接受：", "EmergencyStop accepted by the controller: "),
    ("EmergencyStop 发送失败：", "EmergencyStop transmission failed: "),
    ("HaMeR 实机接管前状态检查失败：", "HaMeR handover precheck failed: "),
    ("HaMeR 实机同步最终检查失败：", "HaMeR Robot Sync final check failed: "),
    ("HaMeR 实机同步未启动：", "HaMeR Robot Sync did not start: "),
    ("HaMeR 实机同步准备失效：", "HaMeR Robot Sync preparation became invalid: "),
    ("当前没有有效手部 cam_t", "no valid hand cam_t is available"),
    ("最终确认后没有有效手部 cam_t", "no valid hand cam_t is available after final confirmation"),
    ("实体 CR3 已离开 Home 或仍在运动", "the physical CR3 left Home or is still moving"),
    ("最终确认期间实体 CR3 离开了 Home", "the physical CR3 left Home during final confirmation"),
    ("最终确认后的手部结果已过期", "the hand result after final confirmation is stale"),
    ("有效手部数据超过", "valid hand data has not updated for more than "),
    ("未更新，实机发送已停止。", "; robot transmission stopped."),
    ("关闭 HaMeR 30003 失败：", "Failed to close HaMeR port 30003: "),
    ("HaMeR 实机同步已中止：", "HaMeR Robot Sync aborted: "),
    ("HaMeR 原点视频设置无效：", "Invalid HaMeR origin-video setting: "),
    ("HaMeR 实时设置无效：", "Invalid HaMeR live setting: "),
    ("HaMeR 启动失败：", "HaMeR startup failed: "),
    ("HaMeR 停止异常：", "HaMeR stop error: "),
    ("HaMeR 视频释放异常：", "HaMeR video release error: "),
    ("Quest 接收启动失败：", "Quest receiver startup failed: "),
    ("Quest 腕部原点已设定：", "Quest wrist origin set: "),
    ("Quest Home 保留腕部原点；", "Quest Home keeps the wrist origin; "),
    ("Quest 实机同步最终检查失败：", "Quest Robot Sync final check failed: "),
    ("Quest 实机同步未启动：", "Quest Robot Sync did not start: "),
    ("Quest 实机同步已中止：", "Quest Robot Sync aborted: "),
    ("Quest 实机接管前状态检查失败：", "Quest handover precheck failed: "),
    ("Quest 实机原点已自动重建：", "Quest robot origin rebuilt automatically: "),
    ("Quest 实机限速已更新并立即生效：", "Quest robot speed limit updated immediately: "),
    ("Quest 第一次确认完成：已按 30004 自动对齐，正在以 ", "First Quest confirmation accepted: aligned from 30004 and returning to shared Home at "),
    (" 回统一 Home。", " to shared Home."),
    ("关闭 Quest 30003 失败：", "Failed to close Quest port 30003: "),
    ("实体 CR3 和 MuJoCo 已到统一 Home；仿真验证偏移已丢弃。", "The physical CR3 and MuJoCo reached shared Home; the simulation-validation offset was discarded. "),
    ("下一帧有效 Quest 腕部位置将自动成为新的实机控制原点。", "The next valid Quest wrist pose will become the new robot-control origin."),
    ("Quest 实机同步准备失效：", "Quest Robot Sync preparation became invalid: "),
    ("Quest 腕部数据超过 ", "Quest wrist data has not updated for more than "),
    ("用户取消了 Quest 实机接管流程。", "The user cancelled the Quest robot handover."),
    ("用户停止了 Quest 实机同步。", "The user stopped Quest Robot Sync."),
    ("Quest 腕部数据不新鲜", "Quest wrist data is stale"),
    ("当前没有新鲜 Quest 腕部数据", "no fresh Quest wrist data is available"),
    ("最终确认后没有 Quest 腕部数据", "no Quest wrist data is available after final confirmation"),
    ("最终确认后的 Quest 腕部数据已过期", "Quest wrist data is stale after final confirmation"),
    ("Quest 实机限速", "Quest robot speed limit"),
    ("示教拖拽要求机器人无报警、已使能且处于空闲模式 5；当前 ", "Teaching requires an alarm-free, enabled robot in idle mode 5; current "),
    ("正在监听 ", "listening on "),
    ("，控制手=", ", control hand="),
    ("无法验证示教前状态：", "Could not verify pre-teaching state: "),
    ("StartDrag 已接受：", "StartDrag accepted: "),
    ("StartDrag 失败：", "StartDrag failed: "),
    ("StopDrag 已接受：", "StopDrag accepted: "),
    ("StopDrag 失败：", "StopDrag failed: "),
    ("请勿假定机械臂已经退出拖拽状态。", "Do not assume the robot has exited drag mode."),
    ("30004 反馈尚未恢复，正在重新连接：", "Port 30004 feedback has not recovered; reconnecting: "),
    ("实时同步停止：", "Live sync stopped: "),
    ("实时同步已启动；整个键盘和画面下方按钮现在控制实体 CR3。", "Live sync started; the keyboard and viewport buttons now control the physical CR3."),
    ("实时同步限速已更新并立即生效：", "Live-sync speed limit updated immediately: "),
    ("MuJoCo 速度", "MuJoCo speed"),
    ("平移步长", "translation step"),
    ("旋转步长", "rotation step"),
    ("同步限速", "live-sync speed limit"),
    ("必须是正的有限数", "must be positive and finite"),
    ("必须是数字", "must be numeric"),
    ("必须是整数", "must be an integer"),
    ("必须在 ", "must be between "),
    (" 到 ", " and "),
    (" 之间", ""),
    ("MuJoCo 更新失败：", "MuJoCo update failed: "),
    ("MuJoCo 渲染失败：", "MuJoCo rendering failed: "),
    ("成功：", " succeeded: "),
    ("失败：", " failed: "),
    ("正在发送", "Sending "),
    ("控制器", "controller"),
    ("伺服", "servo"),
    ("报警", "alarm"),
    ("。", "."),
)


def translate_runtime_log(message: str) -> str:
    """Translate a completed runtime log message while preserving its data."""
    exact = LOG_TEXT_EN.get(message)
    if exact is not None:
        return exact
    translated = message
    for chinese, english in LOG_FRAGMENT_EN:
        translated = translated.replace(chinese, english)
    return translated


NUMPAD_DIRECTIONS = {
    "KP_8": (2, 1.0),
    "KP_Up": (2, 1.0),
    "KP_2": (2, -1.0),
    "KP_Down": (2, -1.0),
    "KP_4": (1, -1.0),
    "KP_Left": (1, -1.0),
    "KP_6": (1, 1.0),
    "KP_Right": (1, 1.0),
    "KP_7": (0, -1.0),
    "KP_Home": (0, -1.0),
    "KP_9": (0, 1.0),
    "KP_Prior": (0, 1.0),
    "KP_1": (3, -1.0),
    "KP_End": (3, -1.0),
    "KP_3": (3, 1.0),
    "KP_Next": (3, 1.0),
    "KP_0": (4, -1.0),
    "KP_Insert": (4, -1.0),
    "KP_Decimal": (4, 1.0),
    "KP_Delete": (4, 1.0),
    "KP_Subtract": (5, -1.0),
    "KP_Add": (5, 1.0),
}

FULL_KEYBOARD_DIRECTIONS = {
    # Translation: letters.
    "w": (2, 1.0),
    "s": (2, -1.0),
    "a": (1, -1.0),
    "d": (1, 1.0),
    "q": (0, -1.0),
    "e": (0, 1.0),
    # Rotation: letters.
    "i": (3, 1.0),
    "k": (3, -1.0),
    "j": (4, -1.0),
    "l": (4, 1.0),
    "u": (5, -1.0),
    "o": (5, 1.0),
    # Translation: navigation keys.
    "Up": (2, 1.0),
    "Down": (2, -1.0),
    "Left": (1, -1.0),
    "Right": (1, 1.0),
    "Prior": (0, 1.0),
    "Next": (0, -1.0),
    # Main number row mirrors the original numpad layout.
    "8": (2, 1.0),
    "2": (2, -1.0),
    "4": (1, -1.0),
    "6": (1, 1.0),
    "7": (0, -1.0),
    "9": (0, 1.0),
    "1": (3, -1.0),
    "3": (3, 1.0),
    "0": (4, -1.0),
    "period": (4, 1.0),
    "minus": (5, -1.0),
    "plus": (5, 1.0),
    "equal": (5, 1.0),
}


def numpad_twist(
    keysym: str,
    *,
    translation_step_m: float,
    rotation_step_rad: float,
) -> np.ndarray | None:
    """Build a Cartesian increment from a Tk numpad keysym."""
    direction = NUMPAD_DIRECTIONS.get(keysym)
    if direction is None:
        return None
    axis, sign = direction
    result = np.zeros(6)
    result[axis] = sign * (
        translation_step_m if axis < 3 else rotation_step_rad
    )
    return result


def keyboard_twist(
    keysym: str,
    *,
    translation_step_m: float,
    rotation_step_rad: float,
) -> np.ndarray | None:
    """Build a Cartesian increment from numpad or full-keyboard controls."""
    direction = NUMPAD_DIRECTIONS.get(keysym)
    if direction is None:
        direction = FULL_KEYBOARD_DIRECTIONS.get(keysym)
    if direction is None:
        direction = FULL_KEYBOARD_DIRECTIONS.get(keysym.lower())
    if direction is None:
        return None
    axis, sign = direction
    result = np.zeros(6)
    result[axis] = sign * (
        translation_step_m if axis < 3 else rotation_step_rad
    )
    return result


def render_size_for_canvas(width: int, height: int) -> tuple[int, int]:
    """Return the exact usable canvas resolution for MuJoCo rendering."""
    return max(MIN_RENDER_WIDTH, int(width)), max(MIN_RENDER_HEIGHT, int(height))


def limit_cartesian_tracking_step(
    error_xyz,
    *,
    max_speed_m_s: float,
    dt: float,
) -> np.ndarray:
    """Rate-limit a Cartesian correction using elapsed wrist-sample time."""
    error = np.asarray(error_xyz, dtype=float).reshape(3)
    if not np.isfinite(error).all():
        raise ValueError("Cartesian tracking error must be finite")
    if not np.isfinite(max_speed_m_s) or max_speed_m_s <= 0.0:
        raise ValueError("Cartesian tracking speed must be positive and finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Cartesian tracking dt must be positive and finite")
    bounded_dt = min(float(dt), QUEST_MAX_CONTROL_DT_S)
    max_step = float(max_speed_m_s) * bounded_dt
    norm = float(np.linalg.norm(error))
    if norm <= max_step or norm == 0.0:
        return error.copy()
    return error * (max_step / norm)


def describe_alarm_groups(
    groups: list[list[int]], language: str = "zh"
) -> list[str]:
    """Resolve GetErrorID groups through the SDK alarm catalog."""
    controller_items, servo_items = alarmAlarmJsonFile()
    controller = {int(item["id"]): item for item in controller_items}
    servo = {int(item["id"]): item for item in servo_items}
    english = language == "en"
    locale = "en" if english else "zh_CN"
    descriptions: list[str] = []
    for group_index, alarm_ids in enumerate(groups):
        catalog = controller if group_index == 0 else servo
        if group_index == 0:
            kind = "Controller" if english else "控制器"
        else:
            kind = f"Servo J{group_index}" if english else f"伺服 J{group_index}"
        for alarm_id in alarm_ids:
            if alarm_id == 0:
                continue
            if alarm_id == -2:
                collision = "Robot collision" if english else "机器人碰撞"
                descriptions.append(f"{kind} [-2] {collision}")
                continue
            item = catalog.get(alarm_id)
            if item is None:
                unknown = "Unknown alarm" if english else "未知报警"
                descriptions.append(f"{kind} [{alarm_id}] {unknown}")
                continue
            localized = item.get(locale, {})
            description = localized.get("description") or (
                "No description provided" if english else "未提供说明"
            )
            solution = localized.get("solution") or ""
            text = f"{kind} [{alarm_id}] {description}"
            if solution:
                text += (
                    f"; Recommendation: {solution}"
                    if english
                    else f"；建议：{solution}"
                )
            descriptions.append(text)
    return descriptions


class RoundedButton(tk.Canvas):
    """Small dependency-free rounded button for the primary GUI controls."""

    def __init__(
        self,
        parent,
        *,
        text: str,
        command=None,
        on_press=None,
        on_release=None,
        fill: str = "#18304d",
        hover_fill: str = "#24527d",
        pressed_fill: str = "#087fb9",
        foreground: str = "#eef6ff",
        height: int = 44,
        font=("Segoe UI Semibold", 11),
    ) -> None:
        super().__init__(
            parent,
            height=height,
            bg="#111c2e",
            highlightthickness=0,
            bd=0,
            cursor="hand2",
            takefocus=False,
        )
        self._text = text
        self._command = command
        self._on_press = on_press
        self._on_release = on_release
        self._fill = fill
        self._hover_fill = hover_fill
        self._pressed_fill = pressed_fill
        self._foreground = foreground
        self._font = font
        self._state = "normal"
        self.bind("<Configure>", self._redraw)
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)

    def set_text(self, text: str) -> None:
        self._text = text
        self._redraw()

    def _current_fill(self) -> str:
        if self._state == "pressed":
            return self._pressed_fill
        if self._state == "hover":
            return self._hover_fill
        return self._fill

    def _redraw(self, _event=None) -> None:
        self.delete("all")
        width = max(4, self.winfo_width())
        height = max(4, self.winfo_height())
        radius = min(14, max(6, height // 3))
        points = (
            2 + radius, 2,
            width - 2 - radius, 2,
            width - 2, 2,
            width - 2, 2 + radius,
            width - 2, height - 2 - radius,
            width - 2, height - 2,
            width - 2 - radius, height - 2,
            2 + radius, height - 2,
            2, height - 2,
            2, height - 2 - radius,
            2, 2 + radius,
            2, 2,
        )
        self.create_polygon(
            points,
            smooth=True,
            splinesteps=24,
            fill=self._current_fill(),
            outline="",
        )
        self.create_text(
            width // 2,
            height // 2,
            text=self._text,
            fill=self._foreground,
            font=self._font,
        )

    def _enter(self, _event) -> None:
        if self._state != "pressed":
            self._state = "hover"
            self._redraw()

    def _leave(self, _event) -> None:
        was_pressed = self._state == "pressed"
        self._state = "normal"
        self._redraw()
        if was_pressed and self._on_release is not None:
            self._on_release()

    def _press(self, _event) -> None:
        self._state = "pressed"
        self._redraw()
        if self._on_press is not None:
            self._on_press()

    def _release(self, event) -> None:
        inside = 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height()
        self._state = "hover" if inside else "normal"
        self._redraw()
        if self._on_release is not None:
            self._on_release()
        if inside and self._command is not None:
            self._command()


class CR3ControlGUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.closed = False
        self.pending_close = False
        self.language = "zh"
        self.log_entries: list[tuple[str, str]] = []
        self.ui_events: queue.Queue[tuple] = queue.Queue()

        self.model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
        self.data = mujoco.MjData(self.model)
        (
            self.arm_joint_ids,
            self.arm_qpos_indices,
            self.arm_dof_indices,
        ) = core.joint_indices(self.model)
        self.end_effector_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "Link6"
        )
        if self.end_effector_id < 0:
            raise ValueError("MuJoCo end-effector body Link6 was not found")

        self.q_target = core.HOME_Q_RAD.copy()
        self.data.qpos[self.arm_qpos_indices] = self.q_target
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.model.vis.global_.offwidth = VIEW_WIDTH
        self.model.vis.global_.offheight = VIEW_HEIGHT
        self.renderer = mujoco.Renderer(
            self.model, height=VIEW_HEIGHT, width=VIEW_WIDTH
        )
        self.render_width = VIEW_WIDTH
        self.render_height = VIEW_HEIGHT
        self.pending_render_size = (VIEW_WIDTH, VIEW_HEIGHT)
        self.render_resize_job: str | None = None
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self._camera_defaults = (
            self.camera.lookat.copy(),
            float(self.camera.distance),
            float(self.camera.azimuth),
            float(self.camera.elevation),
        )

        self.recorder = TrajectoryRecorder(sample_period=1.0 / core.RECORD_RATE_HZ)
        self.saved_path: Path | None = None
        self.dry_run_passed = False
        self.armed_until = 0.0
        self.real_busy = False
        self.real_stop_event = threading.Event()
        self.manual_home_active = False
        self.disable_in_progress = False
        self.estop_latched = False
        self.monitor: FeedbackReceiver | None = None
        self.connected_robot_ip: str | None = None
        self.feedback_connecting_ip: str | None = None
        self.robot_ip_history: list[str] = list(
            dict.fromkeys((args.robot_ip, *KNOWN_ROBOT_IPS))
        )
        self.live_hardware: LiveServoHardware | None = None
        self.live_starting = False
        self.teach_active = False
        self.teach_starting = False
        self.dashboard_action_busy = False
        self.hamer_tracker: HamerBridgeTracker | None = None
        self.hamer_starting = False
        self.hamer_mapper = HamerArmMapper()
        self.hamer_phase = "idle"
        self.hamer_last_sequence = 0
        self.hamer_next_poll = 0.0
        self.hamer_latest_frame: np.ndarray | None = None
        self.hamer_latest_frame_sequence = 0
        self.hamer_rendered_frame_key: tuple[int, int, int] | None = None
        self.hamer_last_status = ""
        self.hamer_real_stage = "idle"
        self.hamer_real_confirm_until = 0.0
        self.hamer_last_valid_hand_at = 0.0
        self.hamer_real_origin_after_sequence = 0
        self.quest_receiver: QuestHandReceiver | None = None
        self.quest_mapper = QuestWristMapper()
        self.quest_orientation_target: np.ndarray | None = None
        self.quest_phase = "idle"
        self.quest_last_sequence = 0
        self.quest_last_wrist_received_at = 0.0
        self.quest_next_poll = 0.0
        self.quest_last_control_time = 0.0
        self.quest_last_wrist_received_at = 0.0
        self.quest_real_stage = "idle"
        self.quest_real_confirm_until = 0.0
        self.quest_real_origin_after_sequence = 0
        self.quest_last_valid_wrist_at = 0.0

        self.last_tick = time.monotonic()
        self.last_render = 0.0
        self.drag_button = 0
        self.drag_x = 0
        self.drag_y = 0
        self.canvas_image_id: int | None = None
        self.photo_image: ImageTk.PhotoImage | None = None
        self.hamer_preview_image_id: int | None = None
        self.hamer_preview_photo: ImageTk.PhotoImage | None = None
        self.pressed_motion_keys: dict[str, float] = {}
        self.key_release_jobs: dict[str, str] = {}

        self.robot_ip_var = tk.StringVar(value=args.robot_ip)
        self.last_robot_ip_text = args.robot_ip
        self.robot_connection_var = tk.StringVar(
            value="未连接 · 输入 IP 后点击“连接 / 切换”"
        )
        self.mode_var = tk.StringVar(value="record")
        self.sim_speed_var = tk.StringVar(value="1.0")
        self.translation_mm_var = tk.StringVar(value="10.0")
        self.rotation_deg_var = tk.StringVar(value="2.0")
        self.real_speed_var = tk.StringVar(value="10")
        self.real_acc_var = tk.StringVar(value="5")
        self.global_speed_var = tk.StringVar(value="10")
        self.live_speed_var = tk.StringVar(
            value=f"{DEFAULT_LIVE_JOINT_SPEED_DEG_S:g}"
        )
        self.status_var = tk.StringVar(value="SIMULATION READY")
        self.robot_status_var = tk.StringVar(value="实机反馈未连接")
        self.joint_feedback_var = tk.StringVar(value="J1—J6：等待实机反馈")
        self.tcp_feedback_var = tk.StringVar(value="TCP：等待实机反馈")
        self.io_feedback_var = tk.StringVar(value="控制器遥测：等待实机反馈")
        self.torque_feedback_var = tk.StringVar(value="关节力学：等待实机反馈")
        self.force_feedback_var = tk.StringVar(value="TCP / 六维力：等待实机反馈")
        self.arm_status_var = tk.StringVar(value="未武装")
        self.keyboard_status_var = tk.StringVar(
            value="键盘待命：W/S=Z  A/D=Y  Q/E=X  I/K=RX  J/L=RY  U/O=RZ"
        )
        self.render_resolution_var = tk.StringVar(value=f"{VIEW_WIDTH} × {VIEW_HEIGHT}")
        self.live_speed_status_var = tk.StringVar(value="实时同步未启动")
        self.teach_status_var = tk.StringVar(value="示教拖拽未启动")
        self.hamer_status_var = tk.StringVar(value="Testing HaMeR=测试视频 · 实时摄像头中按 R 设定原点")
        self.hamer_bridge_url_var = tk.StringVar(value=args.hamer_bridge_url)
        self.hamer_video_var = tk.StringVar(value=args.hamer_video or "")
        self.hamer_camera_id_var = tk.StringVar(value=str(args.hamer_camera_id))
        self.hamer_x_gain_var = tk.StringVar(value="0.03")
        self.hamer_yz_gain_var = tk.StringVar(value="2.0")
        self.hamer_depth_deadband_var = tk.StringVar(value="0.015")
        self.hamer_filter_alpha_var = tk.StringVar(value="0.15")
        self.hamer_filter_deadzone_var = tk.StringVar(value="0.03")
        self.hamer_filter_step_var = tk.StringVar(value="0.06")
        self.hamer_ee_step_var = tk.StringVar(value="0.03")
        self.quest_protocol_var = tk.StringVar(value=args.quest_protocol)
        self.quest_host_var = tk.StringVar(value=args.quest_host)
        self.quest_port_var = tk.StringVar(value=str(args.quest_port))
        self.quest_hand_var = tk.StringVar(value=args.quest_hand)
        self.quest_gain_var = tk.StringVar(value="1.0")
        self.quest_deadzone_mm_var = tk.StringVar(value="1.0")
        self.quest_slow_alpha_var = tk.StringVar(
            value=f"{QUEST_DEFAULT_FILTER_ALPHA:g}"
        )
        self.quest_fast_alpha_var = tk.StringVar(
            value=f"{QUEST_DEFAULT_FILTER_ALPHA:g}"
        )
        self.quest_cartesian_speed_var = tk.StringVar(
            value=f"{QUEST_DEFAULT_CARTESIAN_SPEED_M_S:g}"
        )
        self.quest_real_speed_var = tk.StringVar(
            value=f"{HAMER_REAL_HOME_SPEED_DEG_S:g}"
        )
        self.quest_status_var = tk.StringVar(value="Quest 未启动 · 仅控制 MuJoCo")
        self.last_applied_live_speed: float | None = None

        self._build_ui()
        self.robot_ip_var.trace_add("write", self._on_robot_ip_edited)
        self._bind_events()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(UI_PERIOD_MS, self._tick)
        self.log("自定义 MuJoCo GUI 已启动；当前没有发送实机命令。")
        if not args.enable_real_execution:
            self.log("实机运动锁定：启动时需要 --enable-real-execution。")
        else:
            self.root.after(200, self.connect_feedback)

    def _build_ui(self) -> None:
        self.root.title("CR3 MuJoCo 数字孪生控制")
        self.root.geometry("1650x960")
        self.root.minsize(1320, 820)
        self.root.configure(bg="#0b1220")

        style = ttk.Style(self.root)
        self.style = style
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#0b1220")
        style.configure(
            "TLabel",
            background="#111c2e",
            foreground="#d8e3f0",
            font=("Microsoft YaHei UI", 12),
        )
        style.configure(
            "Header.TLabel",
            background="#0b1220",
            foreground="#f5f8fc",
            font=("Microsoft YaHei UI", 20, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background="#111c2e",
            foreground="#72e3ad",
            font=("Segoe UI Semibold", 12),
            padding=(12, 8),
        )
        style.configure(
            "Card.TLabelframe",
            background="#111c2e",
            bordercolor="#263853",
            lightcolor="#263853",
            darkcolor="#263853",
            relief="solid",
        )
        style.configure(
            "Card.TLabelframe.Label",
            background="#111c2e",
            foreground="#8fbfff",
            font=("Microsoft YaHei UI", 13, "bold"),
        )
        style.configure("Card.TFrame", background="#111c2e")
        style.configure("Card.TLabel", background="#111c2e", foreground="#d8e3f0")
        style.configure(
            "Telemetry.TLabel",
            background="#0d1726",
            foreground="#b9d7ff",
            font=("Microsoft YaHei UI", 12),
            padding=(12, 5),
        )
        style.configure(
            "TButton",
            background="#1b2a41",
            foreground="#eef5ff",
            bordercolor="#365071",
            lightcolor="#1b2a41",
            darkcolor="#1b2a41",
            padding=(13, 10),
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        style.map(
            "TButton",
            background=[("active", "#27466f"), ("pressed", "#16365f")],
            foreground=[("disabled", "#708096")],
        )
        style.configure(
            "Motion.TButton",
            background="#13233a",
            foreground="#b9d7ff",
            padding=(10, 9),
            font=("Segoe UI Semibold", 12),
        )
        style.map("Motion.TButton", background=[("active", "#205b8f"), ("pressed", "#0d87c7")])
        style.configure(
            "TEntry",
            fieldbackground="#0d1726",
            foreground="#f2f7ff",
            font=("Microsoft YaHei UI", 12),
            padding=5,
        )
        style.configure(
            "TCombobox",
            fieldbackground="#0d1726",
            background="#1b2a41",
            foreground="#f2f7ff",
            arrowcolor="#8fbfff",
            font=("Microsoft YaHei UI", 12),
            padding=5,
        )
        style.configure(
            "TSpinbox",
            fieldbackground="#0d1726",
            foreground="#f2f7ff",
            font=("Microsoft YaHei UI", 12),
            padding=5,
        )
        style.configure(
            "TRadiobutton",
            background="#111c2e",
            foreground="#d8e3f0",
            font=("Microsoft YaHei UI", 12),
        )
        style.map("TRadiobutton", background=[("active", "#111c2e")])

        top = ttk.Frame(self.root, padding=(16, 10))
        top.pack(fill=tk.X)
        ttk.Label(top, text="CR3  DIGITAL TWIN", style="Header.TLabel").pack(
            side=tk.LEFT
        )
        ttk.Label(top, textvariable=self.status_var, style="Status.TLabel").pack(
            side=tk.LEFT, padx=(16, 0)
        )
        self.language_button = RoundedButton(
            top,
            text="ENGLISH",
            command=self.toggle_language,
            fill="#17466d",
            hover_fill="#21699f",
            pressed_fill="#0d87c7",
            height=40,
            font=("Segoe UI Semibold", 11),
        )
        self.language_button.configure(width=104)
        self.language_button.pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Label(top, textvariable=self.robot_status_var, style="Status.TLabel").pack(
            side=tk.RIGHT
        )

        body = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        viewport_frame = ttk.Frame(body, style="Card.TFrame", padding=2)
        viewport_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.canvas = tk.Canvas(
            viewport_frame,
            width=VIEW_WIDTH,
            # This is only the requested UI height.  The off-screen renderer
            # is rebuilt to the actual canvas size after layout.  Keeping the
            # request modest lets the controls win when vertical space is
            # limited.
            height=420,
            bg="#111318",
            highlightthickness=0,
            cursor="fleur",
            takefocus=True,
        )
        viewport_status = ttk.Frame(viewport_frame, style="Card.TFrame", padding=(8, 5))
        ttk.Label(
            viewport_status,
            textvariable=self.keyboard_status_var,
            style="Card.TLabel",
        ).pack(side=tk.LEFT)
        ttk.Label(
            viewport_status,
            textvariable=self.render_resolution_var,
            style="Card.TLabel",
        ).pack(side=tk.RIGHT)
        telemetry = ttk.Frame(viewport_frame, style="Card.TFrame", padding=(8, 4))
        self.telemetry_labels = []
        joint_feedback_label = ttk.Label(
            telemetry,
            textvariable=self.joint_feedback_var,
            style="Telemetry.TLabel",
            anchor="w",
            justify=tk.LEFT,
        )
        joint_feedback_label.pack(fill=tk.X, pady=(0, 2))
        self.telemetry_labels.append(joint_feedback_label)
        tcp_feedback_label = ttk.Label(
            telemetry,
            textvariable=self.tcp_feedback_var,
            style="Telemetry.TLabel",
            anchor="w",
            justify=tk.LEFT,
        )
        tcp_feedback_label.pack(fill=tk.X, pady=2)
        self.telemetry_labels.append(tcp_feedback_label)
        io_feedback_label = ttk.Label(
            telemetry,
            textvariable=self.io_feedback_var,
            style="Telemetry.TLabel",
            anchor="w",
            justify=tk.LEFT,
        )
        io_feedback_label.pack(fill=tk.X, pady=2)
        self.telemetry_labels.append(io_feedback_label)
        torque_feedback_label = ttk.Label(
            telemetry,
            textvariable=self.torque_feedback_var,
            style="Telemetry.TLabel",
            anchor="w",
            justify=tk.LEFT,
        )
        torque_feedback_label.pack(fill=tk.X, pady=2)
        self.telemetry_labels.append(torque_feedback_label)
        force_feedback_label = ttk.Label(
            telemetry,
            textvariable=self.force_feedback_var,
            style="Telemetry.TLabel",
            anchor="w",
            justify=tk.LEFT,
        )
        force_feedback_label.pack(fill=tk.X, pady=(2, 0))
        self.telemetry_labels.append(force_feedback_label)
        mouse_help = ttk.Label(
            viewport_frame,
            text="鼠标：左键旋转  ·  右键平移  ·  滚轮缩放",
            style="Card.TLabel",
        )
        self.motion_bar = ttk.LabelFrame(
            viewport_frame,
            text="笛卡尔点动 · 按住连续移动，松开停止",
            padding=7,
            style="Card.TLabelframe",
        )
        motion_buttons = (
            ("X−  [Q]", "q", 0, 0), ("X+  [E]", "e", 1, 0),
            ("Y−  [A]", "a", 0, 1), ("Y+  [D]", "d", 1, 1),
            ("Z−  [S]", "s", 0, 2), ("Z+  [W]", "w", 1, 2),
            ("RX− [K]", "k", 0, 3), ("RX+ [I]", "i", 1, 3),
            ("RY− [J]", "j", 0, 4), ("RY+ [L]", "l", 1, 4),
            ("RZ− [U]", "u", 0, 5), ("RZ+ [O]", "o", 1, 5),
        )
        for column in range(6):
            self.motion_bar.columnconfigure(column, weight=1)
        for label, key, row, column in motion_buttons:
            button = RoundedButton(
                self.motion_bar,
                text=label,
                on_press=lambda k=key: self._start_motion_key(k),
                on_release=lambda k=key: self._stop_motion_key(k),
                height=46,
            )
            button.grid(row=row, column=column, sticky="ew", padx=2, pady=2)

        # Pack fixed controls from the bottom upward before giving the canvas
        # the remaining cavity.  Tk's packer otherwise lets the large canvas
        # request crowd out controls that are packed after it.
        self.motion_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        mouse_help.pack(side=tk.BOTTOM, fill=tk.X, padx=8)
        telemetry.pack(side=tk.BOTTOM, fill=tk.X)
        viewport_status.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Controls deliberately get a wider, two-column area.  A single tall
        # column pushed the lower trajectory buttons outside normal laptop
        # windows.  The viewport is the flexible region and yields space first.
        controls = ttk.Frame(body, width=780)
        controls.grid(row=0, column=1, sticky="ns")
        controls.grid_propagate(False)
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)

        connection = ttk.LabelFrame(
            controls, text="实机连接与安全", padding=10, style="Card.TLabelframe"
        )
        connection.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=(0, 7))
        for column in range(3):
            connection.columnconfigure(column, weight=1)
        ttk.Label(connection, text="目标 IP（可输入或选择）").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        self.robot_ip_combo = ttk.Combobox(
            connection,
            textvariable=self.robot_ip_var,
            values=tuple(self.robot_ip_history),
            width=22,
            state="normal",
        )
        self.robot_ip_combo.grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(4, 0)
        )
        ttk.Label(
            connection,
            textvariable=self.robot_connection_var,
            style="Card.TLabel",
            wraplength=340,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(5, 0))
        self.feedback_connect_button = ttk.Button(
            connection, text="连接 / 切换", command=self.connect_feedback
        )
        self.feedback_connect_button.grid(row=3, column=0, sticky="ew", pady=(7, 0))
        self.feedback_disconnect_button = ttk.Button(
            connection, text="断开反馈", command=self.disconnect_feedback
        )
        self.feedback_disconnect_button.grid(
            row=3, column=1, sticky="ew", padx=5, pady=(7, 0)
        )
        ttk.Button(connection, text="仿真对齐实机", command=self.align_sim_to_real).grid(
            row=3, column=2, sticky="ew", pady=(7, 0)
        )
        ttk.Button(connection, text="① 机器人上电", command=self.on_power_on).grid(
            row=4, column=0, sticky="ew", pady=(6, 0)
        )
        ttk.Button(connection, text="② 使能", command=self.on_enable).grid(
            row=4, column=1, sticky="ew", padx=5, pady=(6, 0)
        )
        ttk.Button(connection, text="取消使能  [Esc]", command=self.on_disable).grid(
            row=4, column=2, sticky="ew", pady=(6, 0)
        )
        ttk.Button(connection, text="读取报警", command=self.on_read_alarms).grid(
            row=5, column=0, sticky="ew", pady=(6, 0)
        )
        ttk.Button(connection, text="清除报警", command=self.on_clear_error).grid(
            row=5, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=(6, 0)
        )
        ttk.Button(connection, text="暂停队列", command=self.on_pause_queue).grid(
            row=6, column=0, sticky="ew", pady=(6, 0)
        )
        ttk.Button(connection, text="继续队列", command=self.on_continue_queue).grid(
            row=6, column=1, columnspan=2, sticky="ew", padx=(5, 0), pady=(6, 0)
        )
        self.real_home_button = ttk.Button(
            connection,
            text="实机回 Home",
            command=self.toggle_real_home,
        )
        self.real_home_button.grid(
            row=7, column=0, columnspan=3, sticky="ew", pady=(7, 0)
        )
        self.estop_button = RoundedButton(
            connection,
            text="软件紧急停止  EMERGENCY STOP",
            command=self.on_emergency_stop,
            fill="#c92a2a",
            hover_fill="#e03131",
            pressed_fill="#9c1c1c",
            height=54,
            font=("Microsoft YaHei UI", 13, "bold"),
        )
        self.estop_button.grid(
            row=8, column=0, columnspan=3, sticky="ew", pady=(8, 2)
        )
        ttk.Label(
            connection,
            text="软件急停不能替代实体急停按钮；解除急停/清报警请在 DobotStudio 手动完成。",
            wraplength=315,
            foreground="#9c1c1c",
        ).grid(row=9, column=0, columnspan=3, sticky="w")
        ttk.Separator(connection, orient=tk.HORIZONTAL).grid(
            row=10, column=0, columnspan=3, sticky="ew", pady=(12, 9)
        )
        ttk.Label(
            connection,
            text=(
                "推荐顺序：① 输入/选择 IP  ② 连接/切换  ③ 上电  ④ 使能  ⑤ 对齐或选择控制模式\n"
                "切换机械臂：先停止运动并建议取消使能，再输入新 IP 并点击“连接 / 切换”"
            ),
            style="Card.TLabel",
            wraplength=350,
            justify=tk.LEFT,
            foreground="#8fbfff",
        ).grid(row=11, column=0, columnspan=3, sticky="nw")

        mode = ttk.LabelFrame(controls, text="控制模式", padding=10, style="Card.TLabelframe")
        mode.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=(0, 7))
        ttk.Radiobutton(
            mode,
            text="录制后回放",
            variable=self.mode_var,
            value="record",
            command=self.on_mode_changed,
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            mode,
            text="实时同步",
            variable=self.mode_var,
            value="live",
            command=self.on_mode_changed,
        ).grid(row=0, column=1, sticky="w", padx=(18, 0))
        ttk.Radiobutton(
            mode,
            text="示教拖拽",
            variable=self.mode_var,
            value="teach",
            command=self.on_mode_changed,
        ).grid(row=0, column=2, sticky="w", padx=(18, 0))
        hamer_mode = ttk.Frame(mode, style="Card.TFrame")
        hamer_mode.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(7, 0))
        for column in range(2):
            hamer_mode.columnconfigure(column, weight=1)
        ttk.Radiobutton(
            hamer_mode,
            text="HaMeR 仿真",
            variable=self.mode_var,
            value="hamer",
            command=self.on_mode_changed,
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            hamer_mode,
            text="HaMeR 参数…",
            command=self.open_hamer_settings,
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.hamer_button = ttk.Button(
            hamer_mode,
            text="HaMeR 实时仿真",
            command=self.toggle_hamer,
        )
        self.hamer_button.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=(5, 0))
        self.hamer_origin_button = ttk.Button(
            hamer_mode,
            text="Testing HaMeR",
            command=self.setup_hamer_origin_video,
        )
        self.hamer_origin_button.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.hamer_stop_video_button = ttk.Button(
            hamer_mode,
            text="停止测试视频",
            command=self.stop_hamer_test_video,
            state=tk.DISABLED,
        )
        self.hamer_stop_video_button.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(5, 0),
        )
        self.hamer_real_preview_button = ttk.Button(
            hamer_mode,
            text="HaMeR 实机同步",
            command=self.toggle_hamer_real_sync,
        )
        self.hamer_real_preview_button.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(5, 0),
        )
        quest_mode = ttk.Frame(mode, style="Card.TFrame")
        quest_mode.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        quest_mode.columnconfigure(0, weight=1)
        quest_mode.columnconfigure(1, weight=1)
        ttk.Radiobutton(
            quest_mode,
            text="Meta Quest 仿真",
            variable=self.mode_var,
            value="quest",
            command=self.on_mode_changed,
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(
            quest_mode,
            text="Quest 设置…",
            command=self.open_quest_settings,
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.quest_button = ttk.Button(
            quest_mode,
            text="启动 Quest 接收",
            command=self.toggle_quest,
        )
        self.quest_button.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.quest_origin_button = ttk.Button(
            quest_mode,
            text="设定腕部原点  [R]",
            command=self.set_quest_origin,
        )
        self.quest_origin_button.grid(
            row=1, column=1, sticky="ew", padx=(5, 0), pady=(5, 0)
        )
        ttk.Label(
            quest_mode,
            text="Quest 实机限速 (deg/s)",
        ).grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.quest_speed_spinbox = ttk.Spinbox(
            quest_mode,
            textvariable=self.quest_real_speed_var,
            from_=0.1,
            to=35.9,
            increment=0.5,
            width=8,
        )
        self.quest_speed_spinbox.grid(
            row=2, column=1, sticky="ew", padx=(5, 0), pady=(5, 0)
        )
        self.quest_real_button = ttk.Button(
            quest_mode,
            text="Quest 实机同步",
            command=self.toggle_quest_real_sync,
        )
        self.quest_real_button.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(5, 0),
        )
        mode_actions = ttk.Frame(mode, style="Card.TFrame")
        mode_actions.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(7, 0))
        mode_actions.columnconfigure(0, weight=1)
        mode_actions.columnconfigure(1, weight=1)
        self.live_button = ttk.Button(
            mode_actions, text="启动实时同步", command=self.toggle_live
        )
        self.live_button.grid(row=0, column=0, sticky="ew")
        self.teach_button = ttk.Button(
            mode_actions, text="进入示教拖拽", command=self.toggle_teach
        )
        self.teach_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        ttk.Label(
            mode,
            textvariable=self.live_speed_status_var,
            style="Card.TLabel",
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(7, 0))
        ttk.Label(
            mode,
            textvariable=self.teach_status_var,
            style="Card.TLabel",
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(3, 0))
        ttk.Label(
            mode,
            textvariable=self.hamer_status_var,
            style="Card.TLabel",
            wraplength=315,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(3, 0))
        ttk.Label(
            mode,
            textvariable=self.quest_status_var,
            style="Card.TLabel",
            wraplength=315,
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(3, 0))

        parameters = ttk.LabelFrame(controls, text="参数调节", padding=10, style="Card.TLabelframe")
        parameters.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=(0, 7))
        parameter_rows = (
            ("MuJoCo 倍速", self.sim_speed_var, 0.1, 5.0, 0.1),
            ("平移 (mm)", self.translation_mm_var, 0.1, 50.0, 0.5),
            ("旋转 (deg)", self.rotation_deg_var, 0.1, 10.0, 0.5),
            ("回放 SpeedJ", self.real_speed_var, 1, 19, 1),
            ("回放 AccJ（保留）", self.real_acc_var, 1, 19, 1),
            ("同步限速", self.live_speed_var, 0.1, 35.9, 0.5),
            ("全局倍率", self.global_speed_var, 1, 19, 1),
        )
        for index, (label, variable, low, high, step) in enumerate(parameter_rows):
            row = index // 2
            column = (index % 2) * 2
            ttk.Label(parameters, text=label).grid(row=row, column=column, sticky="w")
            ttk.Spinbox(
                parameters,
                textvariable=variable,
                from_=low,
                to=high,
                increment=step,
                width=6,
            ).grid(row=row, column=column + 1, sticky="e", padx=(5, 8), pady=2)
        parameters.columnconfigure(0, weight=1)
        parameters.columnconfigure(2, weight=1)
        parameter_button_row = math.ceil(len(parameter_rows) / 2)
        ttk.Button(
            parameters,
            text="应用实机全局倍率",
            command=self.on_apply_global_speed,
        ).grid(row=parameter_button_row, column=0, columnspan=4, sticky="ew", pady=(6, 0))

        trajectory = ttk.LabelFrame(controls, text="仿真与轨迹", padding=10, style="Card.TLabelframe")
        trajectory.grid(row=1, column=1, sticky="nsew", padx=(4, 0), pady=(0, 7))
        for column in range(2):
            trajectory.columnconfigure(column, weight=1)
        self.record_button = ttk.Button(
            trajectory, text="开始记录  [KP Enter]", command=self.toggle_record
        )
        self.record_button.grid(row=0, column=0, sticky="ew")
        ttk.Button(trajectory, text="仿真 Home  [KP 5]", command=self.go_home).grid(
            row=0, column=1, sticky="ew", padx=(5, 0)
        )
        ttk.Button(trajectory, text="清空轨迹  [KP /]", command=self.clear_trajectory).grid(
            row=1, column=0, sticky="ew", pady=(5, 0)
        )
        ttk.Button(trajectory, text="Dry Run  [KP *]", command=self.run_dry_review).grid(
            row=1, column=1, sticky="ew", padx=(5, 0), pady=(5, 0)
        )
        ttk.Button(trajectory, text="武装真实回放", command=self.arm_real_playback).grid(
            row=2, column=0, sticky="ew", pady=(5, 0)
        )
        self.confirm_button = ttk.Button(
            trajectory, text="最终确认执行", command=self.confirm_real_playback
        )
        self.confirm_button.grid(
            row=2, column=1, sticky="ew", padx=(5, 0), pady=(5, 0)
        )
        ttk.Button(trajectory, text="重置相机", command=self.reset_camera).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        ttk.Label(trajectory, textvariable=self.arm_status_var).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(5, 0)
        )

        log_frame = ttk.LabelFrame(controls, text="运行日志", padding=7, style="Card.TLabelframe")
        log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        controls.rowconfigure(2, weight=1)
        log_toolbar = ttk.Frame(log_frame, style="Card.TFrame")
        log_toolbar.pack(fill=tk.X, pady=(0, 5))
        self.clear_log_button = ttk.Button(
            log_toolbar,
            text="清空日志",
            command=self.clear_log,
        )
        self.clear_log_button.pack(side=tk.RIGHT)
        self.log_text = tk.Text(
            log_frame,
            height=10,
            width=46,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Microsoft YaHei UI", 11),
            bg="#09111d",
            fg="#a9c1d9",
            insertbackground="#ffffff",
            selectbackground="#1f5f91",
            relief=tk.FLAT,
            padx=8,
            pady=6,
            spacing1=2,
            spacing3=3,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _tr(self, chinese: str, english: str | None = None) -> str:
        if self.language == "zh":
            return chinese
        if english is not None:
            return english
        return UI_TEXT_EN.get(chinese, chinese)

    def _set_button_text(self, widget, chinese: str) -> None:
        text = self._tr(chinese)
        if isinstance(widget, RoundedButton):
            widget.set_text(text)
        else:
            widget.configure(text=text)

    def _walk_widgets(self, parent):
        for child in parent.winfo_children():
            yield child
            yield from self._walk_widgets(child)

    def _apply_language_fonts(self) -> None:
        ui_font = "Microsoft YaHei UI" if self.language == "zh" else "Segoe UI"
        self.style.configure("TLabel", font=(ui_font, 12))
        self.style.configure("Header.TLabel", font=(ui_font, 20, "bold"))
        self.style.configure("Status.TLabel", font=("Segoe UI Semibold", 12))
        self.style.configure(
            "Card.TLabelframe.Label", font=(ui_font, 13, "bold")
        )
        self.style.configure("Telemetry.TLabel", font=(ui_font, 12))
        self.style.configure("TButton", font=(ui_font, 12, "bold"))
        self.style.configure("TRadiobutton", font=(ui_font, 12))
        self.style.configure("TEntry", font=(ui_font, 12))
        self.style.configure("TCombobox", font=(ui_font, 12))
        self.style.configure("TSpinbox", font=(ui_font, 12))
        self.log_text.configure(font=(ui_font, 11))

    def _apply_language_to_widgets(self) -> None:
        translations = UI_TEXT_EN if self.language == "en" else UI_TEXT_ZH
        for widget in self._walk_widgets(self.root):
            if isinstance(widget, RoundedButton):
                continue
            try:
                current = str(widget.cget("text"))
            except (tk.TclError, AttributeError):
                continue
            translated = translations.get(current)
            if translated is not None:
                widget.configure(text=translated)

        runtime_variables = (
            self.robot_status_var,
            self.robot_connection_var,
            self.joint_feedback_var,
            self.tcp_feedback_var,
            self.io_feedback_var,
            self.torque_feedback_var,
            self.force_feedback_var,
            self.arm_status_var,
            self.keyboard_status_var,
            self.live_speed_status_var,
            self.teach_status_var,
            self.hamer_status_var,
            self.quest_status_var,
        )
        for variable in runtime_variables:
            current = variable.get()
            translated = translations.get(current)
            if translated is not None:
                variable.set(translated)

        self.language_button.set_text("中文" if self.language == "en" else "ENGLISH")
        self.estop_button.set_text(
            self._tr("软件紧急停止  EMERGENCY STOP")
        )
        self.root.title(
            "CR3 MuJoCo Sim2Real Control"
            if self.language == "en"
            else "CR3 MuJoCo 数字孪生控制"
        )
        self._apply_language_fonts()

    def toggle_language(self) -> None:
        self.language = "en" if self.language == "zh" else "zh"
        self._apply_language_to_widgets()
        self._render_log_entries()
        source = None
        if self.live_hardware is not None:
            source = self.live_hardware.feedback
        elif self.monitor is not None:
            source = self.monitor
        if source is not None:
            try:
                self._update_feedback_display(source.require_fresh(max_age=0.75))
            except Exception:
                pass
        self._on_robot_ip_edited()
        self.log(
            "界面语言已切换为英文。"
            if self.language == "en"
            else "界面语言已切换为中文。"
        )

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", lambda event: self._drag_start(event, 1))
        self.canvas.bind("<ButtonPress-3>", lambda event: self._drag_start(event, 3))
        self.canvas.bind("<B1-Motion>", self._drag_motion)
        self.canvas.bind("<B3-Motion>", self._drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)
        self.canvas.bind("<ButtonRelease-3>", self._drag_end)
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Configure>", self._canvas_configured)
        self.canvas.bind("<Button-1>", self._focus_canvas, add="+")
        self.root.bind_all("<KeyPress>", self._key_press, add="+")
        self.root.bind_all("<KeyRelease>", self._key_release, add="+")
        self.root.bind("<FocusOut>", self._window_focus_out, add="+")

    def _focus_canvas(self, _event=None) -> None:
        self.canvas.focus_set()

    def _canvas_configured(self, event) -> None:
        telemetry_wrap = max(420, event.width - 36)
        for label in self.telemetry_labels:
            label.configure(wraplength=telemetry_wrap)
        desired = render_size_for_canvas(event.width, event.height)
        self.pending_render_size = desired
        self.render_resolution_var.set(f"调整中 → {desired[0]} × {desired[1]}")
        if self.render_resize_job is not None:
            self.root.after_cancel(self.render_resize_job)
        self.render_resize_job = self.root.after(
            RENDER_RESIZE_DEBOUNCE_MS, self._rebuild_renderer_for_canvas
        )

    def _rebuild_renderer_for_canvas(self) -> None:
        self.render_resize_job = None
        width, height = self.pending_render_size
        if (width, height) == (self.render_width, self.render_height):
            self.render_resolution_var.set(f"{width} × {height} · 1:1")
            return
        old_width, old_height = self.render_width, self.render_height
        old_renderer = self.renderer
        old_renderer.close()
        try:
            self.model.vis.global_.offwidth = width
            self.model.vis.global_.offheight = height
            new_renderer = mujoco.Renderer(self.model, height=height, width=width)
        except Exception as exc:
            self.log(f"无法将渲染分辨率调整为 {width}×{height}：{exc}")
            self.model.vis.global_.offwidth = old_width
            self.model.vis.global_.offheight = old_height
            self.renderer = mujoco.Renderer(
                self.model, height=old_height, width=old_width
            )
            self.render_resolution_var.set(
                f"{self.render_width} × {self.render_height} · fallback"
            )
            return
        self.renderer = new_renderer
        self.render_width = width
        self.render_height = height
        self.render_resolution_var.set(f"{width} × {height} · 1:1")
        self.last_render = 0.0

    def _drag_start(self, event, button: int) -> None:
        self.canvas.focus_set()
        self.drag_button = button
        self.drag_x = event.x
        self.drag_y = event.y

    def _drag_motion(self, event) -> None:
        dx = event.x - self.drag_x
        dy = event.y - self.drag_y
        self.drag_x = event.x
        self.drag_y = event.y
        if self.drag_button == 1:
            self.camera.azimuth -= dx * 0.35
            self.camera.elevation = float(
                np.clip(self.camera.elevation - dy * 0.35, -89.0, 89.0)
            )
        elif self.drag_button == 3:
            azimuth = math.radians(self.camera.azimuth)
            scale = self.camera.distance * 0.0015
            right = np.array([math.cos(azimuth), -math.sin(azimuth), 0.0])
            up = np.array([0.0, 0.0, 1.0])
            self.camera.lookat[:] += (-dx * right + dy * up) * scale

    def _drag_end(self, _event) -> None:
        self.drag_button = 0

    def _mouse_wheel(self, event) -> None:
        factor = math.exp(-event.delta / 1200.0)
        self.camera.distance = float(
            np.clip(self.camera.distance * factor, 0.15, 20.0)
        )

    def reset_camera(self) -> None:
        lookat, distance, azimuth, elevation = self._camera_defaults
        self.camera.lookat[:] = lookat
        self.camera.distance = distance
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation

    def _key_press(self, event) -> str | None:
        if event.keysym == "Escape":
            self._clear_motion_keys()
            self.on_disable(confirm=False, force=True)
            return "break"
        if self._is_parameter_editor(event.widget):
            self.keyboard_status_var.set(
                self._tr("正在编辑参数；Esc=强制取消使能")
            )
            return None
        try:
            twist = keyboard_twist(
                event.keysym,
                translation_step_m=self._translation_step_m(),
                rotation_step_rad=self._rotation_step_rad(),
            )
        except ValueError as exc:
            self.log(f"参数无效：{exc}")
            return "break"
        if twist is not None:
            self._start_motion_key(event.keysym, twist=twist)
            return "break"
        if event.keysym in ("KP_5", "KP_Begin"):
            self.go_home()
            return "break"
        if event.keysym == "KP_Enter":
            if self.recorder.recording:
                self.stop_recording()
            elif self.recorder.review_ready and self.dry_run_passed:
                self.arm_real_playback()
            else:
                self.start_recording()
            return "break"
        if event.keysym == "Return":
            if event.widget.winfo_class() in {"Button", "TButton"}:
                return None
            self.toggle_record()
            return "break"
        if event.keysym == "KP_Divide":
            self.clear_trajectory()
            return "break"
        if event.keysym == "KP_Multiply":
            if self.armed_until > 0.0:
                self.confirm_real_playback()
            else:
                self.run_dry_review()
            return "break"
        if event.keysym.lower() == "r":
            if self.mode_var.get() == "hamer":
                self.set_hamer_camera_origin()
            elif self.mode_var.get() == "quest":
                self.set_quest_origin()
            else:
                self.toggle_record()
            return "break"
        if event.keysym.lower() == "c":
            self.clear_trajectory()
            return "break"
        if event.keysym.lower() == "v":
            self.run_dry_review()
            return "break"
        if event.keysym.lower() == "h":
            self.go_home()
            return "break"
        return None

    def _key_release(self, event) -> str | None:
        key = self._canonical_motion_key(event.keysym)
        if key not in self.pressed_motion_keys:
            return None
        previous_job = self.key_release_jobs.pop(key, None)
        if previous_job is not None:
            self.root.after_cancel(previous_job)

        def finish_release():
            self.key_release_jobs.pop(key, None)
            self._stop_motion_key(key)

        self.key_release_jobs[key] = self.root.after(35, finish_release)
        return "break"

    @staticmethod
    def _canonical_motion_key(keysym: str) -> str:
        return keysym.lower() if len(keysym) == 1 else keysym

    def _start_motion_key(self, keysym: str, twist: np.ndarray | None = None) -> None:
        key = self._canonical_motion_key(keysym)
        release_job = self.key_release_jobs.pop(key, None)
        if release_job is not None:
            self.root.after_cancel(release_job)
        if key in self.pressed_motion_keys:
            return
        if twist is None:
            try:
                twist = keyboard_twist(
                    key,
                    translation_step_m=self._translation_step_m(),
                    rotation_step_rad=self._rotation_step_rad(),
                )
            except ValueError as exc:
                self.log(f"参数无效：{exc}")
                return
        if twist is None or not self.apply_twist(twist):
            return
        self.pressed_motion_keys[key] = time.monotonic() + KEY_HOLD_DELAY_S
        self.keyboard_status_var.set(
            self._tr(
                f"控制中：{key}（松开即停止）",
                f"Controlling: {key} (release to stop)",
            )
        )

    def _stop_motion_key(self, keysym: str) -> None:
        key = self._canonical_motion_key(keysym)
        self.pressed_motion_keys.pop(key, None)
        release_job = self.key_release_jobs.pop(key, None)
        if release_job is not None:
            try:
                self.root.after_cancel(release_job)
            except Exception:
                pass
        if not self.pressed_motion_keys:
            self.keyboard_status_var.set(
                self._tr("键盘待命：W/S=Z  A/D=Y  Q/E=X  I/K=RX  J/L=RY  U/O=RZ")
            )

    def _clear_motion_keys(self) -> None:
        self.pressed_motion_keys.clear()
        for job in self.key_release_jobs.values():
            try:
                self.root.after_cancel(job)
            except Exception:
                pass
        self.key_release_jobs.clear()

    def _window_focus_out(self, _event) -> None:
        self.root.after(20, self._clear_keys_if_window_inactive)

    def _clear_keys_if_window_inactive(self) -> None:
        # ttk.Combobox creates a transient ``popdown`` widget.  During its
        # teardown Tk can return a focus path that no longer exists in
        # ``root.children``; tkinter's focus_get() then raises KeyError while
        # resolving that stale path.  Treat this short race as an unknown
        # focus state and wait for the next FocusOut event instead of letting
        # the periodic callback fail.
        try:
            focus = self.root.focus_get()
        except (KeyError, tk.TclError):
            return
        if focus is None:
            self._clear_motion_keys()
            self.keyboard_status_var.set(self._tr("窗口失焦：运动键已清除"))

    def _repeat_motion_keys(self, now: float) -> None:
        due_keys = [
            key for key, next_time in self.pressed_motion_keys.items()
            if now >= next_time
        ]
        if not due_keys:
            return
        combined = np.zeros(6)
        try:
            for key in due_keys:
                twist = keyboard_twist(
                    key,
                    translation_step_m=self._translation_step_m(),
                    rotation_step_rad=self._rotation_step_rad(),
                )
                if twist is not None:
                    combined += twist
                self.pressed_motion_keys[key] = now + KEY_REPEAT_PERIOD_S
        except ValueError as exc:
            self._clear_motion_keys()
            self.log(f"参数无效：{exc}")
            return
        if np.any(combined) and not self.apply_twist(combined):
            self._clear_motion_keys()

    @staticmethod
    def _is_parameter_editor(widget) -> bool:
        try:
            widget_class = widget.winfo_class()
        except Exception:
            return False
        return widget_class in {
            "Entry",
            "TEntry",
            "Spinbox",
            "TSpinbox",
            "TCombobox",
            "Text",
        }

    def apply_twist(self, twist: np.ndarray) -> bool:
        if self.real_busy or self.estop_latched:
            self.log("控制已锁定：真实回放中或软件急停已锁存。")
            return False
        if self.mode_var.get() == "live" and self.live_hardware is None:
            self.log("请先点击“启动实时同步”。")
            return False
        if self.mode_var.get() == "teach" or self.teach_active:
            self.log("示教拖拽模式禁用键盘和 GUI 点动，请直接安全地拖动实体机械臂。")
            return False
        if self.mode_var.get() == "hamer":
            self.log("HaMeR 模式由手部位移独占控制；请停止 HaMeR 实时仿真或切换模式后再键盘点动。")
            return False
        if self.mode_var.get() == "quest":
            self.log("Quest 模式由腕部位移独占控制；按 R 可重设原点，或停止 Quest 接收后再点动。")
            return False
        self.q_target = core.apply_cartesian_increment(
            self.model,
            self.data,
            self.end_effector_id,
            self.arm_dof_indices,
            self.arm_joint_ids,
            self.q_target,
            twist,
        )
        return True

    def go_home(self) -> None:
        if self.mode_var.get() in ("live", "teach"):
            self.log("实时同步或示教拖拽模式禁用 Home。")
            return
        if self.hamer_real_stage != "idle" or self.quest_real_stage != "idle":
            self.log("手部实机接管流程中禁用手动 Home；请先停止实机同步。")
            return
        if self.real_busy or self.estop_latched:
            return
        self.q_target = core.HOME_Q_RAD.copy()
        if self.mode_var.get() == "hamer" and self.hamer_mapper.calibrated:
            home_data = mujoco.MjData(self.model)
            home_data.qpos[:] = self.data.qpos
            home_data.qpos[self.arm_qpos_indices] = core.HOME_Q_RAD
            home_data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, home_data)
            home_ee_pos = home_data.xpos[self.end_effector_id].copy()
            self.hamer_mapper.reanchor_robot_origin(home_ee_pos)
            self.hamer_status_var.set("HaMeR 手部原点已保留 · 机械臂基准已重新锚定到 Home")
            self.log(
                "Home 不清除 HaMeR 手部原点；"
                f"Link6 基准已更新为 {np.round(home_ee_pos, 4).tolist()}。"
            )
        elif self.mode_var.get() == "quest" and self.quest_mapper.calibrated:
            home_data = mujoco.MjData(self.model)
            home_data.qpos[:] = self.data.qpos
            home_data.qpos[self.arm_qpos_indices] = core.HOME_Q_RAD
            home_data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, home_data)
            home_ee_pos = home_data.xpos[self.end_effector_id].copy()
            self.quest_orientation_target = home_data.xmat[
                self.end_effector_id
            ].reshape(3, 3).copy()
            self.quest_mapper.reanchor_robot_origin(home_ee_pos)
            self.quest_status_var.set("Quest 腕部原点已保留 · Link6 基准已对齐 Home")
            self.log(
                "Quest Home 保留腕部原点；"
                f"Link6 基准已更新为 {np.round(home_ee_pos, 4).tolist()}。"
            )
        self.log("MuJoCo 正在返回 Home；实体机器人不受影响。")

    def toggle_real_home(self) -> None:
        """Start or cancel a guarded low-speed physical Home move."""
        if self.manual_home_active:
            self.real_stop_event.set()
            self._set_button_text(self.real_home_button, "正在停止实机 Home…")
            self.log("正在停止实机回 Home；不会继续发送新的 ServoJ 目标。")
            return
        self.start_real_home()

    def start_real_home(self) -> None:
        if not self.args.enable_real_execution:
            self.log("实机回 Home 已锁定：请使用 --enable-real-execution 启动 GUI。")
            return
        if self.estop_latched:
            self.log("软件急停已锁存；请在 DobotStudio 清除并重新使能。")
            return
        if self.dashboard_action_busy:
            self.log("另一个机器人状态命令正在执行，请稍候。")
            return
        if (
            self.real_busy
            or self.live_hardware is not None
            or self.live_starting
            or self.teach_active
            or self.teach_starting
            or self.hamer_real_stage != "idle"
            or self.quest_real_stage != "idle"
        ):
            self.log("当前存在其他实体运动流程；停止后才能让实机回 Home。")
            return
        if not MAPPING_CALIBRATED:
            self.log("关节映射未标定，禁止实体 CR3 回 Home。")
            return
        if self.monitor is None:
            self.log("正在先连接 30004 只读反馈；连接后请再次点击“实机回 Home”。")
            self.connect_feedback()
            return
        robot_ip = self._active_robot_ip()
        if robot_ip is None:
            return
        try:
            state = self.monitor.require_fresh(max_age=0.5)
            require_motion_ready(state)
        except Exception as exc:
            self.log(f"实机回 Home 前状态检查失败：{exc}")
            return

        home_deg = sim_rad_to_real_deg(core.HOME_Q_RAD)
        max_error = float(np.max(np.abs(state.joints_deg - home_deg)))
        if not messagebox.askyesno(
            self._tr("实体 CR3 回 Home", "Physical CR3 Home"),
            self._tr(
                "即将让实体 CR3 低速返回统一 Home。\n\n"
                f"IP: {robot_ip}\n"
                f"当前关节: {np.round(state.joints_deg, 2).tolist()}°\n"
                f"目标 Home: {np.round(home_deg, 2).tolist()}°\n"
                f"最大关节差: {max_error:.2f}°\n"
                f"主机限速: {MANUAL_REAL_HOME_SPEED_DEG_S:.1f}°/s\n\n"
                "请确认机械臂到 Home 的完整路径和周围工作区安全。",
                "The physical CR3 will return to shared Home at low speed.\n\n"
                f"IP: {robot_ip}\n"
                f"Current joints: {np.round(state.joints_deg, 2).tolist()}°\n"
                f"Target Home: {np.round(home_deg, 2).tolist()}°\n"
                f"Maximum joint difference: {max_error:.2f}°\n"
                f"Host speed limit: {MANUAL_REAL_HOME_SPEED_DEG_S:.1f}°/s\n\n"
                "Confirm the complete path to Home and the surrounding workspace are safe.",
            ),
            icon="warning",
        ):
            self.log("用户取消了实机回 Home。")
            return

        self.real_busy = True
        self.manual_home_active = True
        self.real_stop_event = threading.Event()
        self.q_target = core.HOME_Q_RAD.copy()
        self._set_button_text(self.real_home_button, "停止实机回 Home")
        self.status_var.set("REAL ROBOT HOMING")
        self.log(
            "实体 CR3 正在低速回 Home："
            f"IP={robot_ip}，限速={MANUAL_REAL_HOME_SPEED_DEG_S:.1f}°/s。"
        )

        def work():
            hardware = LiveServoHardware(
                robot_ip,
                max_joint_speed_deg_s=MANUAL_REAL_HOME_SPEED_DEG_S,
                feedback=self.monitor,
            )
            try:
                hardware.connect()
                return stream_live_target_until_reached(
                    hardware,
                    home_deg,
                    stop_event=self.real_stop_event,
                )
            finally:
                hardware.close()

        self._background(work, self._real_home_complete, self._real_home_failed)

    def _real_home_complete(self, state) -> None:
        self.real_busy = False
        self.manual_home_active = False
        self._set_button_text(self.real_home_button, "实机回 Home")
        home_ee = self._set_sim_home_now()
        if self.quest_mapper.calibrated and hasattr(self, "data"):
            self.quest_orientation_target = self.data.xmat[
                self.end_effector_id
            ].reshape(3, 3).copy()
        if self.hamer_mapper.calibrated:
            self.hamer_mapper.reanchor_robot_origin(home_ee)
        if self.quest_mapper.calibrated:
            self.quest_mapper.reanchor_robot_origin(home_ee)
        self._update_feedback_display(state)
        self.status_var.set("REAL ROBOT AT HOME")
        self.log("实体 CR3 已到 Home；MuJoCo 已对齐，已有手部原点已重新锚定到 Link6。")

    def _real_home_failed(self, exc: Exception) -> None:
        was_stopping = self.real_stop_event.is_set()
        self.real_busy = False
        self.manual_home_active = False
        self._set_button_text(self.real_home_button, "实机回 Home")
        if self.monitor is not None:
            try:
                state = self.monitor.require_fresh(max_age=0.75)
                self._set_sim_from_real(state.joints_deg)
            except Exception:
                pass
        if self.estop_latched or self.disable_in_progress:
            self.log(f"实机回 Home 已由停止命令中止：{exc}")
            return
        if was_stopping:
            self.status_var.set("REAL ROBOT HOMING CANCELLED")
            self.log("实机回 Home 已由用户停止。")
        else:
            self.status_var.set("REAL ROBOT HOMING FAILED")
            self.log(f"实机回 Home 失败：{exc}")

    def toggle_record(self) -> None:
        if self.recorder.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        mode = self.mode_var.get()
        teach_recording = mode == "teach" and self.teach_active
        if (mode != "record" and not teach_recording) or self.real_busy:
            self.log("只有仿真录制模式，或已验证的示教拖拽模式可以记录轨迹。")
            return
        now = time.monotonic()
        self.recorder.start(
            now,
            self.data.qpos[self.arm_qpos_indices],
            self.data.xpos[self.end_effector_id],
        )
        self.saved_path = None
        self.dry_run_passed = False
        self.armed_until = 0.0
        self._set_button_text(self.record_button, "停止记录  [KP Enter]")
        self.status_var.set("TEACH RECORDING" if teach_recording else "RECORDING")
        self.log("示教轨迹记录已开始。" if teach_recording else "轨迹记录已开始。")

    def stop_recording(self) -> None:
        if not self.recorder.recording:
            return
        now = time.monotonic()
        self.recorder.stop(
            now,
            self.data.qpos[self.arm_qpos_indices],
            self.data.xpos[self.end_effector_id],
        )
        self.saved_path = self.recorder.save_json(self.args.output_dir.resolve())
        self._set_button_text(self.record_button, "开始记录  [KP Enter]")
        self.status_var.set(
            "TEACH TRAJECTORY READY" if self.teach_active else "TRAJECTORY READY"
        )
        self.log(
            f"记录完成：{len(self.recorder.points)} samples，"
            f"{self.recorder.duration:.3f} s"
        )
        self.log(f"已保存：{self.saved_path}")

    def clear_trajectory(self) -> None:
        if self.real_busy:
            return
        self.recorder.clear()
        self.saved_path = None
        self.dry_run_passed = False
        self.armed_until = 0.0
        self.arm_status_var.set("未武装")
        self._set_button_text(self.record_button, "开始记录  [KP Enter]")
        self.status_var.set("SIMULATION READY")
        self.log("轨迹已清空；实体机器人未收到命令。")

    def run_dry_review(self) -> None:
        if not self.recorder.review_ready:
            self.log("没有已停止并可供检查的轨迹。")
            return
        errors = validate_trajectory(
            self.recorder.points, self.model.jnt_range[self.arm_joint_ids]
        )
        if errors:
            self.dry_run_passed = False
            self.log("轨迹安全检查失败：")
            for error in errors:
                self.log(f"  - {error}")
            return
        try:
            speed, acceleration = self._real_motion_parameters()
            commands = build_servoj_dry_run(
                self.recorder.points,
                sample_period=LIVE_SERVO_PERIOD_S,
            )
        except Exception as exc:
            self.log(f"参数无效：{exc}")
            return
        self.dry_run_passed = True
        self.status_var.set("DRY RUN PASSED")
        self.log(
            f"Dry Run 通过：{len(commands)} 个连续 ServoJ 采样，"
            f"{1 / LIVE_SERVO_PERIOD_S:.1f} Hz，主机限速 "
            f"{CR3_MAX_JOINT_SPEED_DEG_S * speed / 100.0:.1f}°/s；"
            f"AccJ={acceleration}% 仅保留，ServoJ 不使用"
        )
        for command in commands[:5]:
            self.log(command)
        if len(commands) > 5:
            self.log(f"... 还有 {len(commands) - 5} 个 ServoJ 采样")
        self.log("尚未连接 30003，尚未发送运动命令。")

    def arm_real_playback(self) -> None:
        if not self.args.enable_real_execution:
            self.log("实机运动锁定：请使用 --enable-real-execution 启动 GUI。")
            return
        if self.estop_latched:
            self.log("软件急停已锁存；请在 DobotStudio 清除并重新使能。")
            return
        if self.teach_active or self.teach_starting:
            self.log("请先退出示教拖拽，再武装真实回放。")
            return
        if self.mode_var.get() == "hamer" or self.hamer_tracker is not None:
            self.log("HaMeR 当前是仿真专用模式；请停止 HaMeR 实时仿真并切换到录制回放后再武装实机。")
            return
        if self.mode_var.get() == "quest" or self.quest_receiver is not None:
            self.log("Quest 仿真正在运行；请先停止 Quest 接收并切换到录制回放模式。")
            return
        if not self.dry_run_passed or not self.recorder.review_ready:
            self.log("请先完成轨迹记录和 Dry Run。")
            return
        if not MAPPING_CALIBRATED:
            self.log("关节映射未标定，禁止真实回放。")
            return
        self.armed_until = time.monotonic() + core.REAL_CONFIRMATION_WINDOW_S
        self.arm_status_var.set("真实回放已武装：15 秒内点击最终确认")
        self.status_var.set("REAL PLAYBACK ARMED")
        self.log("真实回放已武装 15 秒；尚未发送运动命令。")

    def confirm_real_playback(self) -> None:
        now = time.monotonic()
        if self.armed_until <= 0.0 or now > self.armed_until:
            self.armed_until = 0.0
            self.arm_status_var.set("武装已过期")
            self.log("真实回放武装已过期，请重新武装。")
            return
        if self.real_busy or self.estop_latched or self.teach_active:
            return
        try:
            speed, acceleration = self._real_motion_parameters()
        except ValueError as exc:
            self.log(f"参数无效：{exc}")
            return
        robot_ip = self._active_robot_ip()
        if robot_ip is None:
            return
        if not messagebox.askyesno(
            "最终真实运动确认",
            "即将控制实体 CR3。\n\n"
            f"IP: {robot_ip}\n"
            f"主机关节限速: {CR3_MAX_JOINT_SPEED_DEG_S * speed / 100.0:.1f} deg/s\n"
            f"AccJ: {acceleration}%（保留参数，ServoJ 不使用）\n\n"
            "确认工作区安全并执行吗？",
            icon="warning",
        ):
            self.log("用户取消了最终真实回放确认。")
            return

        self.armed_until = 0.0
        self.arm_status_var.set("真实回放执行中")
        self.real_busy = True
        self.real_stop_event = threading.Event()
        self.status_var.set("REAL PLAYBACK RUNNING")
        self.log("最终确认完成，正在连接并执行低速真实回放。")

        def work():
            core.execute_real_trajectory(
                self.recorder,
                robot_ip=robot_ip,
                sdk_root=core.ROOT,
                speed_percent=speed,
                acceleration_percent=acceleration,
                stop_event=self.real_stop_event,
                feedback=self.monitor,
            )

        self._background(work, self._playback_complete, self._playback_failed)

    def _playback_complete(self, _result=None) -> None:
        self.real_busy = False
        self.arm_status_var.set("真实回放完成")
        self.status_var.set("REAL PLAYBACK COMPLETE")
        self.log("真实回放已完成。")
        self.align_sim_to_real()

    def _playback_failed(self, exc: Exception) -> None:
        self.real_busy = False
        self.arm_status_var.set("真实回放已中止")
        self.status_var.set("REAL PLAYBACK ABORTED")
        self.log(f"真实回放中止：{exc}")

    @staticmethod
    def _normalize_robot_ip(value: str) -> str:
        text = value.strip()
        try:
            address = ipaddress.ip_address(text)
        except ValueError as exc:
            raise ValueError(f"无效的机器人 IP：{text or '（空）'}") from exc
        if address.version != 4:
            raise ValueError("机器人 IP 必须是 IPv4 地址")
        return str(address)

    def _selected_robot_ip(self) -> str | None:
        try:
            return self._normalize_robot_ip(self.robot_ip_var.get())
        except ValueError as exc:
            self.log(str(exc))
            self.robot_connection_var.set(self._tr("IP 格式无效", "Invalid IP address"))
            return None

    def _active_robot_ip(self) -> str | None:
        """Return the explicitly connected IP, never an unapplied text edit."""
        selected = self._selected_robot_ip()
        if selected is None:
            return None
        if self.monitor is None or self.connected_robot_ip is None:
            self.log("请先输入目标 IP，并点击“连接 / 切换”。")
            return None
        if self.monitor.robot_ip != self.connected_robot_ip:
            self.log("内部连接状态不一致；为安全起见已阻止实机命令，请重新连接反馈。")
            return None
        if selected != self.connected_robot_ip:
            self.log(
                f"目标 IP {selected} 尚未生效；当前连接仍是 "
                f"{self.connected_robot_ip}。请先点击“连接 / 切换”。"
            )
            return None
        return self.connected_robot_ip

    def _connected_robot_ip_for_stop(self) -> str | None:
        """Use the active robot for stop/disable even after a pending IP edit."""
        if self.monitor is None or self.connected_robot_ip is None:
            self.log("当前没有已连接的机器人。")
            return None
        if self.monitor.robot_ip != self.connected_robot_ip:
            self.log("内部连接状态不一致；请使用实体停止并重新连接反馈。")
            return None
        return self.connected_robot_ip

    def _robot_connection_busy(self) -> bool:
        return bool(
            self.dashboard_action_busy
            or self.real_busy
            or self.live_hardware is not None
            or self.live_starting
            or self.teach_active
            or self.teach_starting
            or self.hamer_real_stage != "idle"
            or self.quest_real_stage != "idle"
        )

    def _on_robot_ip_edited(self, *_args) -> None:
        raw = self.robot_ip_var.get().strip()
        ip_text_changed = raw != self.last_robot_ip_text
        self.last_robot_ip_text = raw
        if ip_text_changed and getattr(self, "armed_until", 0.0) > 0.0:
            self.armed_until = 0.0
            self.arm_status_var.set(self._tr("未武装", "Not armed"))
            self.log("目标 IP 已修改；真实回放武装已取消。")
        try:
            selected = self._normalize_robot_ip(raw)
        except ValueError:
            self.robot_connection_var.set(self._tr("IP 格式无效", "Invalid IP address"))
            return
        if self.feedback_connecting_ip is not None:
            self.robot_connection_var.set(
                self._tr(
                    f"正在连接 {self.feedback_connecting_ip} · 请稍候",
                    f"Connecting to {self.feedback_connecting_ip} · please wait",
                )
            )
        elif self.connected_robot_ip == selected and self.monitor is not None:
            self.robot_connection_var.set(
                self._tr(f"已连接 · {selected}", f"Connected · {selected}")
            )
        elif self.connected_robot_ip is not None:
            self.robot_connection_var.set(
                self._tr(
                    f"待切换到 {selected} · 当前 {self.connected_robot_ip}",
                    f"Pending {selected} · active {self.connected_robot_ip}",
                )
            )
        else:
            self.robot_connection_var.set(
                self._tr(
                    f"待连接 · {selected}",
                    f"Ready to connect · {selected}",
                )
            )

    def connect_feedback(self) -> None:
        robot_ip = self._selected_robot_ip()
        if robot_ip is None:
            return
        if self.feedback_connecting_ip is not None:
            self.log(f"正在连接 {self.feedback_connecting_ip}，请稍候。")
            return
        switching = (
            self.connected_robot_ip is not None
            and self.connected_robot_ip != robot_ip
        )
        if switching and self._robot_connection_busy():
            self.log("当前有实机命令或运动流程正在运行，停止后才能切换机器人 IP。")
            return
        if self.monitor is not None and self.connected_robot_ip == robot_ip:
            try:
                state = self.monitor.require_fresh(max_age=0.75)
            except Exception as exc:
                self.log(f"现有 30004 反馈不可用，正在重新建立连接：{exc}")
                self.monitor.close()
                self.monitor = None
                self.connected_robot_ip = None
            else:
                self.robot_status_var.set(self._format_robot_state(state))
                self.log("30004 只读反馈正常，无需重复连接。")
                return
        elif self.monitor is not None:
            previous_ip = self.connected_robot_ip or self.monitor.robot_ip
            self.monitor.close()
            self.monitor = None
            self.connected_robot_ip = None
            self.log(f"已断开 {previous_ip} 的 30004 反馈；正在切换到 {robot_ip}。")

        self.feedback_connecting_ip = robot_ip
        self.robot_status_var.set(self._tr("正在连接 30004...", "Connecting to port 30004..."))
        self.robot_connection_var.set(
            self._tr(f"正在连接 {robot_ip} · 请稍候", f"Connecting to {robot_ip} · please wait")
        )
        self._set_button_text(self.feedback_connect_button, "正在连接…")

        def work():
            receiver = FeedbackReceiver(robot_ip)
            try:
                state = receiver.start()
                return robot_ip, receiver, state
            except Exception:
                receiver.close()
                raise

        self._background(work, self._feedback_connected, self._feedback_failed)

    def _feedback_connected(self, result) -> None:
        robot_ip, receiver, state = result
        self.feedback_connecting_ip = None
        self._set_button_text(self.feedback_connect_button, "连接 / 切换")
        try:
            selected = self._normalize_robot_ip(self.robot_ip_var.get())
        except ValueError:
            selected = None
        if selected != robot_ip:
            receiver.close()
            self.robot_connection_var.set(
                self._tr("IP 已修改 · 请重新点击连接", "IP changed · connect again")
            )
            self.log(f"{robot_ip} 已连接，但输入 IP 已改变；该连接未启用。")
            return
        if self.monitor is not None:
            self.monitor.close()
        self.monitor = receiver
        self.connected_robot_ip = robot_ip
        if robot_ip not in self.robot_ip_history:
            self.robot_ip_history.append(robot_ip)
            self.robot_ip_combo.configure(values=tuple(self.robot_ip_history))
        self.robot_status_var.set(self._format_robot_state(state))
        self.robot_connection_var.set(
            self._tr(f"已连接 · {robot_ip}", f"Connected · {robot_ip}")
        )
        self._update_feedback_display(state)
        self.log(f"{robot_ip}:30004 只读反馈已连接；后续实机命令已锁定到该 IP。")
        if not self.recorder.recording and self.live_hardware is None:
            self._set_sim_from_real(state.joints_deg)

    def _feedback_failed(self, exc: Exception) -> None:
        failed_ip = self.feedback_connecting_ip
        self.feedback_connecting_ip = None
        self.connected_robot_ip = None
        self._set_button_text(self.feedback_connect_button, "连接 / 切换")
        self.robot_status_var.set("30004 连接失败")
        self.robot_connection_var.set(
            self._tr(
                f"连接失败 · {failed_ip or '未知 IP'}",
                f"Connection failed · {failed_ip or 'unknown IP'}",
            )
        )
        self.joint_feedback_var.set("J1—J6：反馈连接失败")
        self.tcp_feedback_var.set("TCP：反馈连接失败")
        self.io_feedback_var.set("控制器遥测：反馈连接失败")
        self.torque_feedback_var.set("关节力学：反馈连接失败")
        self.force_feedback_var.set("TCP / 六维力：反馈连接失败")
        self.log(f"只读反馈连接失败：{exc}")

    def disconnect_feedback(self) -> None:
        if self.feedback_connecting_ip is not None:
            self.log("反馈连接正在建立，请等待结果后再断开。")
            return
        if self._robot_connection_busy():
            self.log("当前有实机命令或运动流程正在运行，停止后才能断开反馈。")
            return
        if self.monitor is None:
            self.log("当前没有已连接的 30004 反馈。")
            return
        previous_ip = self.connected_robot_ip or self.monitor.robot_ip
        self.monitor.close()
        self.monitor = None
        self.connected_robot_ip = None
        self.robot_status_var.set(self._tr("实机反馈未连接", "Robot feedback disconnected"))
        self.robot_connection_var.set(
            self._tr(f"已断开 · {previous_ip}", f"Disconnected · {previous_ip}")
        )
        self.log(f"已断开 {previous_ip}:30004；不会再向该机器人发送命令。")

    def align_sim_to_real(self) -> None:
        if self._active_robot_ip() is None:
            return
        try:
            state = self.monitor.require_fresh(max_age=0.5)
        except Exception as exc:
            self.log(f"无法读取新鲜实机反馈：{exc}")
            return
        if (
            self.recorder.recording
            or self.real_busy
            or self.hamer_real_stage != "idle"
            or self.quest_real_stage != "idle"
        ):
            self.log("记录或回放过程中不能重新对齐。")
            return
        self._set_sim_from_real(state.joints_deg)
        self.log("MuJoCo 已对齐到实体 CR3 当前关节姿态。")

    def _set_sim_from_real(self, joints_deg) -> None:
        q = real_deg_to_sim_rad(joints_deg)
        self.q_target = q.copy()
        self.data.qpos[self.arm_qpos_indices] = q
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def on_read_alarms(self) -> None:
        if self.dashboard_action_busy:
            self.log("另一个控制器命令正在执行，请稍候。")
            return
        robot_ip = self._active_robot_ip()
        if robot_ip is None:
            return
        self.dashboard_action_busy = True
        self.status_var.set("READING CR3 ALARMS")
        self.log("正在通过 GetErrorID() 读取控制器和六轴伺服报警。")
        self._background(
            lambda: get_robot_error_ids(robot_ip),
            self._alarms_read,
            self._alarm_read_failed,
        )

    def _alarms_read(self, result) -> None:
        self.dashboard_action_busy = False
        groups, _reply = result
        descriptions = describe_alarm_groups(groups, self.language)
        if not descriptions:
            self.status_var.set("NO ACTIVE ALARMS")
            self.log("GetErrorID：当前没有活动报警。")
            return
        self.status_var.set(f"{len(descriptions)} ACTIVE ALARM(S)")
        self.log(f"GetErrorID：发现 {len(descriptions)} 条活动报警：")
        for description in descriptions:
            self.log(f"  • {description}")

    def _alarm_read_failed(self, exc: Exception) -> None:
        self.dashboard_action_busy = False
        self.status_var.set("ALARM READ FAILED")
        self.log(f"读取报警失败：{exc}")

    def on_clear_error(self) -> None:
        if not self.args.enable_real_execution:
            self.log("清除报警已锁定：请使用 --enable-real-execution 启动 GUI。")
            return
        if self.dashboard_action_busy:
            self.log("另一个控制器命令正在执行，请稍候。")
            return
        if self.real_busy or self.live_hardware is not None or self.teach_active:
            self.log("请先停止真实回放、实时同步或示教拖拽，再清除报警。")
            return
        if not messagebox.askyesno(
            "清除 CR3 报警",
            "只会发送 ClearError()，不会自动使能，也不会自动继续运动队列。\n\n"
            "请先排除碰撞、急停或硬件故障原因。",
            icon="warning",
        ):
            return
        robot_ip = self._active_robot_ip()
        if robot_ip is None:
            return
        self.dashboard_action_busy = True
        self.status_var.set("CLEARING CR3 ALARMS")
        self.log("正在发送 ClearError()；不会自动 Continue 或 Enable。")
        self._background(
            lambda: clear_robot_error(robot_ip),
            lambda reply: self._dashboard_command_complete("ClearError", reply),
            lambda exc: self._dashboard_command_failed("ClearError", exc),
        )

    def on_pause_queue(self) -> None:
        if not self.args.enable_real_execution:
            self.log("暂停队列已锁定：请使用 --enable-real-execution 启动 GUI。")
            return
        if self.dashboard_action_busy:
            self.log("另一个控制器命令正在执行，请稍候。")
            return
        if self.teach_active:
            self.log("请先退出示教拖拽，再暂停运动队列。")
            return
        robot_ip = self._connected_robot_ip_for_stop()
        if robot_ip is None:
            return
        self.dashboard_action_busy = True
        self.real_stop_event.set()
        self.armed_until = 0.0
        self.arm_status_var.set("未武装")
        self.live_starting = False
        self.hamer_real_stage = "idle"
        self.hamer_real_confirm_until = 0.0
        self.quest_real_stage = "idle"
        self.quest_real_confirm_until = 0.0
        if hasattr(self, "hamer_real_preview_button"):
            self._set_button_text(self.hamer_real_preview_button, "HaMeR 实机同步")
        if hasattr(self, "quest_real_button"):
            self._set_button_text(self.quest_real_button, "Quest 实机同步")
        live = self.live_hardware
        self.live_hardware = None
        self.last_applied_live_speed = None
        self._set_button_text(self.live_button, "启动实时同步")
        self.live_speed_status_var.set("实时同步未启动")
        self.status_var.set("PAUSING CR3 QUEUE")
        self.log("本地运动发送已停止，正在发送 pause()。")

        def work():
            try:
                return pause_robot_queue(robot_ip)
            finally:
                if live is not None:
                    live.close()

        self._background(
            work,
            lambda reply: self._dashboard_command_complete("pause", reply),
            lambda exc: self._dashboard_command_failed("pause", exc),
        )

    def on_continue_queue(self) -> None:
        if not self.args.enable_real_execution:
            self.log("继续队列已锁定：请使用 --enable-real-execution 启动 GUI。")
            return
        if self.dashboard_action_busy:
            self.log("另一个控制器命令正在执行，请稍候。")
            return
        if (
            self.real_busy
            or self.live_hardware is not None
            or self.live_starting
            or self.teach_active
        ):
            self.log("活动运动流程会自行管理队列，此时不能手动 Continue。")
            return
        if not messagebox.askyesno(
            "继续 CR3 运动队列",
            "Continue() 可能恢复控制器中尚未完成的排队命令。\n\n"
            "请确认机械臂周围安全，并确认需要恢复该队列。",
            icon="warning",
        ):
            return
        robot_ip = self._active_robot_ip()
        if robot_ip is None:
            return
        self.dashboard_action_busy = True
        self.status_var.set("RESUMING CR3 QUEUE")
        self.log("正在发送 continue()。")
        self._background(
            lambda: continue_robot_queue(robot_ip),
            lambda reply: self._dashboard_command_complete("Continue", reply),
            lambda exc: self._dashboard_command_failed("Continue", exc),
        )

    def on_apply_global_speed(self) -> None:
        if not self.args.enable_real_execution:
            self.log("实机全局倍率已锁定：请使用 --enable-real-execution 启动 GUI。")
            return
        if self.dashboard_action_busy:
            self.log("另一个控制器命令正在执行，请稍候。")
            return
        if (
            self.real_busy
            or self.live_hardware is not None
            or self.live_starting
            or self.teach_active
        ):
            self.log("请先停止真实回放或实时同步，再修改实机全局倍率。")
            return
        try:
            percent = self._bounded_int(
                self.global_speed_var, 1, 19, "实机全局倍率"
            )
        except ValueError as exc:
            self.log(f"参数无效：{exc}")
            return
        if not messagebox.askyesno(
            "应用实机全局倍率",
            f"将向控制器发送 SpeedFactor({percent})。\n\n"
            "该倍率作用于控制器后续运动；实时同步仍同时受主机端 deg/s 限速约束。",
            icon="warning",
        ):
            return
        robot_ip = self._active_robot_ip()
        if robot_ip is None:
            return
        self.dashboard_action_busy = True
        self.status_var.set("SETTING GLOBAL SPEED")
        self.log(f"正在发送 SpeedFactor({percent})。")
        self._background(
            lambda: set_robot_speed_factor(robot_ip, percent),
            lambda reply: self._dashboard_command_complete("SpeedFactor", reply),
            lambda exc: self._dashboard_command_failed("SpeedFactor", exc),
        )

    def _dashboard_command_complete(self, command: str, reply: str) -> None:
        self.dashboard_action_busy = False
        self.status_var.set(f"{command.upper()} ACCEPTED")
        self.log(f"{command} 成功：{reply.strip()}")

    def _dashboard_command_failed(self, command: str, exc: Exception) -> None:
        self.dashboard_action_busy = False
        self.status_var.set(f"{command.upper()} FAILED")
        self.log(f"{command} 失败：{exc}")

    def on_enable(self) -> None:
        if not self.args.enable_real_execution:
            self.log("使能已锁定：请使用 --enable-real-execution 启动 GUI。")
            return
        if self.dashboard_action_busy:
            self.log("另一个机器人状态命令正在执行，请稍候。")
            return
        if self.real_busy or self.live_hardware is not None or self.teach_active:
            self.log("实体运动连接活动时不需要重复使能。")
            return
        robot_ip = self._active_robot_ip()
        if robot_ip is None:
            return
        if not messagebox.askyesno(
            "使能实体 CR3",
            f"将向 {robot_ip}:29999 发送 EnableRobot()。\n\n"
            "请确认机械臂周围安全，并且已经进入 TCP/IP 二次开发模式。",
            icon="warning",
        ):
            return
        self.dashboard_action_busy = True
        self.status_var.set("ENABLING REAL CR3")
        self.log("正在发送 EnableRobot()...")
        self._background(
            lambda: enable_robot(robot_ip),
            self._enable_complete,
            self._enable_failed,
        )

    def _enable_complete(self, reply: str) -> None:
        self.dashboard_action_busy = False
        self.estop_latched = False
        self.status_var.set("ENABLE COMMAND ACCEPTED")
        self.log(f"EnableRobot 成功：{reply.strip()}")
        if self.monitor is None:
            self.connect_feedback()

    def _enable_failed(self, exc: Exception) -> None:
        self.dashboard_action_busy = False
        self.status_var.set("ENABLE FAILED")
        self.log(f"EnableRobot 失败：{exc}")

    def on_power_on(self) -> None:
        if not self.args.enable_real_execution:
            self.log("上电已锁定：请使用 --enable-real-execution 启动 GUI。")
            return
        if self.dashboard_action_busy:
            self.log("另一个机器人状态命令正在执行，请稍候。")
            return
        if self.real_busy or self.live_hardware is not None or self.teach_active:
            self.log("实体运动连接活动时不能重复执行上电。")
            return
        robot_ip = self._active_robot_ip()
        if robot_ip is None:
            return
        if not messagebox.askyesno(
            "机器人上电",
            f"将向 {robot_ip}:29999 发送 PowerOn()。\n\n"
            "请确认控制器和机械臂周围安全。上电完成后仍需单独点击使能。",
            icon="warning",
        ):
            return
        self.dashboard_action_busy = True
        self.status_var.set("POWERING ON REAL CR3")
        self.log("正在发送 PowerOn()；请等待末端指示灯状态稳定。")
        self._background(
            lambda: power_on_robot(robot_ip),
            self._power_on_complete,
            self._power_on_failed,
        )

    def _power_on_complete(self, reply: str) -> None:
        self.dashboard_action_busy = False
        self.status_var.set("POWER ON COMMAND ACCEPTED")
        self.log(f"PowerOn 成功：{reply.strip()}")
        self.log("机器人已收到上电命令；状态稳定后再点击“使能 Enable”。")
        if self.monitor is None:
            self.connect_feedback()

    def _power_on_failed(self, exc: Exception) -> None:
        self.dashboard_action_busy = False
        self.status_var.set("POWER ON FAILED")
        self.log(f"PowerOn 失败：{exc}")

    def on_disable(self, *, confirm: bool = True, force: bool = False) -> None:
        if not self.args.enable_real_execution:
            self.log("取消使能已锁定：请使用 --enable-real-execution 启动 GUI。")
            return
        if self.disable_in_progress:
            self.log("DisableRobot() 已在发送中。")
            return
        if self.dashboard_action_busy and not force:
            self.log("另一个机器人状态命令正在执行，请稍候。")
            return
        robot_ip = self._connected_robot_ip_for_stop()
        if robot_ip is None:
            return
        active_motion = (
            self.real_busy
            or self.live_hardware is not None
            or self.hamer_real_stage != "idle"
            or self.quest_real_stage != "idle"
            or self.teach_active
        )
        message = (
            "将停止本地运动发送并向控制器发送 DisableRobot()。\n\n"
            "这是正常停止/下使能，不是急停。"
        )
        if active_motion:
            message += "\n\n当前存在活动的实体运动连接，将立即中止。"
        if confirm:
            if not messagebox.askyesno("取消机器人使能", message, icon="warning"):
                return

        if self.recorder.recording:
            self.stop_recording()
        self._clear_motion_keys()
        self.disable_in_progress = True
        self.dashboard_action_busy = True
        self.real_stop_event.set()
        self.armed_until = 0.0
        self.arm_status_var.set("未武装")
        self.live_starting = False
        self.hamer_real_stage = "idle"
        self.hamer_real_confirm_until = 0.0
        self.quest_real_stage = "idle"
        self.quest_real_confirm_until = 0.0
        if hasattr(self, "hamer_real_preview_button"):
            self._set_button_text(self.hamer_real_preview_button, "HaMeR 实机同步")
        if hasattr(self, "quest_real_button"):
            self._set_button_text(self.quest_real_button, "Quest 实机同步")
        self.teach_active = False
        self.teach_starting = False
        self._set_button_text(self.teach_button, "进入示教拖拽")
        self.teach_status_var.set("示教拖拽未启动")
        live = self.live_hardware
        self.live_hardware = None
        self.last_applied_live_speed = None
        self._set_button_text(self.live_button, "启动实时同步")
        self.live_speed_status_var.set("实时同步未启动")
        self.status_var.set("DISABLING REAL CR3")
        self.log("本地运动发送已停止，正在发送 DisableRobot()。")

        def work():
            try:
                return disable_robot(robot_ip, feedback=self.monitor)
            finally:
                if live is not None:
                    live.close()

        self._background(work, self._disable_complete, self._disable_failed)

    def _disable_complete(self, reply: str) -> None:
        self.disable_in_progress = False
        self.dashboard_action_busy = False
        self.status_var.set("ROBOT DISABLED")
        self.log(f"DisableRobot 成功：{reply.strip()}")
        self.log("机器人已正常取消使能；无需使用软件急停。")

    def _disable_failed(self, exc: Exception) -> None:
        self.disable_in_progress = False
        self.dashboard_action_busy = False
        self.status_var.set("DISABLE FAILED")
        self.log(f"DisableRobot 失败：{exc}")

    def on_emergency_stop(self) -> None:
        if self.estop_latched:
            self.log("软件急停已经锁存。")
            return
        if self.recorder.recording:
            self.stop_recording()
        self.estop_latched = True
        self.real_stop_event.set()
        self.armed_until = 0.0
        self.arm_status_var.set("软件急停已锁存")
        self.status_var.set("SOFTWARE EMERGENCY STOP")
        self.log("软件急停触发：本地运动发送已锁定，正在发送 EmergencyStop()。")

        live = self.live_hardware
        self.live_hardware = None
        self.hamer_real_stage = "idle"
        self.hamer_real_confirm_until = 0.0
        self.quest_real_stage = "idle"
        self.quest_real_confirm_until = 0.0
        if hasattr(self, "hamer_real_preview_button"):
            self._set_button_text(self.hamer_real_preview_button, "HaMeR 实机同步")
        if hasattr(self, "quest_real_button"):
            self._set_button_text(self.quest_real_button, "Quest 实机同步")
        self.teach_active = False
        self.teach_starting = False
        self._set_button_text(self.teach_button, "进入示教拖拽")
        self.teach_status_var.set("示教拖拽已由急停中止")
        self.last_applied_live_speed = None
        self._set_button_text(self.live_button, "启动实时同步")
        self.live_speed_status_var.set("实时同步未启动")
        # An emergency stop always targets the explicitly connected robot,
        # even if the user has typed a different (not yet applied) IP.
        robot_ip = self.connected_robot_ip
        if robot_ip is None:
            robot_ip = self._selected_robot_ip()
        if robot_ip is None:
            self.log("没有有效的实机 IP；软件急停只完成了本地锁存，请使用实体急停。")
            return

        def work():
            try:
                return emergency_stop_robot(robot_ip)
            finally:
                if live is not None:
                    live.close()

        self._background(work, self._estop_complete, self._estop_failed)

    def _estop_complete(self, reply: str) -> None:
        self.log(f"EmergencyStop 已被控制器接受：{reply.strip()}")
        self.log("请在 DobotStudio 手动解除急停、清除报警，然后重新使能。")
        if self.pending_close:
            self._finish_close()

    def _estop_failed(self, exc: Exception) -> None:
        self.log(f"EmergencyStop 发送失败：{exc}")
        messagebox.showerror(
            "软件急停发送失败",
            f"无法确认控制器收到软件急停：\n{exc}\n\n请立即使用实体急停按钮。",
        )
        if self.pending_close:
            self._finish_close()

    def open_hamer_settings(self) -> None:
        """Open the compact origin-video/live-camera mapping editor."""
        window = tk.Toplevel(self.root)
        window.title("HaMeR 仿真控制设置")
        window.transient(self.root)
        window.resizable(False, False)
        panel = ttk.LabelFrame(
            window,
            text="HaMeR 原点视频与相对位移映射",
            padding=14,
            style="Card.TLabelframe",
        )
        panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        fields = (
            ("HaMeR 服务地址", self.hamer_bridge_url_var),
            ("内置原点定义视频", self.hamer_video_var),
            ("实时模式摄像头 ID", self.hamer_camera_id_var),
            ("深度→X 增益", self.hamer_x_gain_var),
            ("水平/垂直→Y/Z 增益", self.hamer_yz_gain_var),
            ("深度死区 (m)", self.hamer_depth_deadband_var),
            ("EMA 系数", self.hamer_filter_alpha_var),
            ("目标死区 (m)", self.hamer_filter_deadzone_var),
            ("滤波最大步长 (m)", self.hamer_filter_step_var),
            ("单帧 IK 最大步长 (m)", self.hamer_ee_step_var),
        )
        entries = []
        for row, (label, variable) in enumerate(fields):
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=3)
            entry = ttk.Entry(panel, textvariable=variable, width=42)
            entry.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=3)
            entries.append(entry)

        def browse_video() -> None:
            selected = filedialog.askopenfilename(
                parent=window,
                title="选择 HaMeR 原点定义视频",
                filetypes=(("视频", "*.avi *.mp4 *.mov *.mkv"), ("所有文件", "*.*")),
            )
            if selected:
                self.hamer_video_var.set(selected)

        ttk.Button(panel, text="更换原点视频…", command=browse_video).grid(
            row=1, column=2, padx=(6, 0), pady=3
        )
        ttk.Label(
            panel,
            text="Testing HaMeR 第一次点击打开测试视频，第二次选当前帧开始验证。实时摄像头中按 R 设置仿真原点；HaMeR 实机同步会自动让仿真和实机回到统一 Home，并在最终确认前重新归零。",
            wraplength=520,
            foreground="#72e3ad",
        ).grid(row=len(fields), column=0, columnspan=3, sticky="w", pady=(10, 4))
        ttk.Button(panel, text="完成", command=window.destroy).grid(
            row=len(fields) + 1, column=0, columnspan=3, sticky="ew", pady=(7, 0)
        )
        panel.columnconfigure(1, weight=1)
        entries[0].focus_set()

    def _new_hamer_mapper(self) -> HamerArmMapper:
        try:
            x_gain = float(self.hamer_x_gain_var.get())
            yz_gain = float(self.hamer_yz_gain_var.get())
            depth_deadband = float(self.hamer_depth_deadband_var.get())
            alpha = float(self.hamer_filter_alpha_var.get())
            target_deadzone = float(self.hamer_filter_deadzone_var.get())
            filter_step = float(self.hamer_filter_step_var.get())
            max_ee_step = float(self.hamer_ee_step_var.get())
        except ValueError as exc:
            raise ValueError("HaMeR 映射参数必须是数字") from exc
        values = np.array(
            [x_gain, yz_gain, depth_deadband, alpha, target_deadzone, filter_step, max_ee_step]
        )
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("HaMeR 映射参数必须是正的有限数")
        if alpha > 1.0:
            raise ValueError("HaMeR EMA 系数必须在 (0, 1] 内")
        return HamerArmMapper(
            depth_scale=x_gain,
            lateral_vertical_scale=yz_gain,
            depth_deadband=depth_deadband,
            ema_alpha=alpha,
            filter_deadzone=target_deadzone,
            filter_max_step=filter_step,
        )

    def toggle_hamer(self) -> None:
        if self.hamer_phase in ("origin_starting", "origin_video", "origin_validation"):
            self.log("原点设置/验证视频正在播放，请等待视频结束。")
            return
        if self.hamer_phase in ("live_starting", "live_wait_origin", "live"):
            self.stop_hamer(clear_origin=True)
            return
        self.start_hamer_live()

    def toggle_hamer_real_sync(self) -> None:
        """Advance or stop the guarded HaMeR simulation-to-real handover."""
        if self.hamer_real_stage == "idle":
            self.prepare_hamer_real_home()
        elif self.hamer_real_stage == "ready":
            self.confirm_hamer_real_sync()
        elif self.hamer_real_stage in ("homing", "waiting_origin", "connecting"):
            self.stop_hamer_real_sync("用户取消了 HaMeR 实机接管流程。")
        elif self.hamer_real_stage == "active":
            self.stop_hamer_real_sync("用户停止了 HaMeR 实机同步。")

    def _set_sim_home_now(self) -> np.ndarray:
        """Set MuJoCo exactly to the shared Home and return Link6 position."""
        self.q_target = core.HOME_Q_RAD.copy()
        self.data.qpos[self.arm_qpos_indices] = core.HOME_Q_RAD
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[self.end_effector_id].copy()

    def prepare_hamer_real_home(self) -> None:
        """First confirmation: align, then move real and simulation to Home."""
        if not self.args.enable_real_execution:
            self.log("HaMeR 实机同步锁定：请使用 --enable-real-execution 启动 GUI。")
            return
        if self.estop_latched or self.real_busy or self.teach_active:
            self.log("软件急停、真实回放或示教拖拽活动时不能启动 HaMeR 实机同步。")
            return
        if self.hamer_phase != "live" or not self.hamer_mapper.calibrated:
            self.log("请先启动 HaMeR 实时仿真，在摄像头中按 R，并完成仿真验证。")
            return
        if self.hamer_tracker is None:
            self.log("HaMeR 摄像头未运行。")
            return
        if self.live_hardware is not None or self.live_starting:
            self.log("另一个实时实机连接已经活动，请先停止。")
            return
        if self.monitor is None:
            self.log("正在先连接 30004 只读反馈；连接后请再次点击 HaMeR 实机同步。")
            self.connect_feedback()
            return
        robot_ip = self._active_robot_ip()
        if robot_ip is None:
            return
        try:
            state = self.monitor.require_fresh(max_age=0.5)
            require_motion_ready(state)
        except Exception as exc:
            self.log(f"HaMeR 实机接管前状态检查失败：{exc}")
            return

        home_deg = sim_rad_to_real_deg(core.HOME_Q_RAD)
        if not messagebox.askyesno(
            "HaMeR 实机同步 · 第一次确认",
            "将冻结 HaMeR 仿真目标，并让实体 CR3 与 MuJoCo 自动回到统一 Home。\n\n"
            f"当前实机关节: {np.round(state.joints_deg, 2).tolist()}°\n"
            f"目标 Home: {np.round(home_deg, 2).tolist()}°\n"
            f"回 Home 限速: {HAMER_REAL_HOME_SPEED_DEG_S:.1f} deg/s\n\n"
            "确认工作区安全并开始低速回 Home 吗？",
            icon="warning",
        ):
            self.log("用户取消了 HaMeR 实机同步第一次确认。")
            return

        # Alignment is part of the handover now; the user no longer has to
        # remember to press the separate alignment button.
        self._set_sim_from_real(state.joints_deg)
        self.real_stop_event = threading.Event()
        self.hamer_real_stage = "homing"
        self.hamer_real_confirm_until = 0.0
        self.q_target = core.HOME_Q_RAD.copy()
        self._set_button_text(self.hamer_real_preview_button, "停止回 Home")
        self.hamer_status_var.set("HaMeR 实机准备 · 仿真与实机正在回统一 Home")
        self.status_var.set("HAMER REAL HOMING")
        self.log("第一次确认完成：已按 30004 自动对齐，正在以 5°/s 回统一 Home。")
        def work():
            hardware = LiveServoHardware(
                robot_ip,
                max_joint_speed_deg_s=HAMER_REAL_HOME_SPEED_DEG_S,
                feedback=self.monitor,
            )
            try:
                hardware.connect()
                return stream_live_target_until_reached(
                    hardware,
                    home_deg,
                    stop_event=self.real_stop_event,
                )
            finally:
                hardware.close()

        self._background(work, self._hamer_real_home_complete, self._hamer_real_failed)

    def _hamer_real_home_complete(self, state) -> None:
        if self.hamer_real_stage != "homing" or self.estop_latched:
            return
        self._set_sim_home_now()
        self.hamer_real_stage = "waiting_origin"
        self.hamer_real_origin_after_sequence = (
            0 if self.hamer_tracker is None else self.hamer_tracker.get().sequence
        )
        self._set_button_text(self.hamer_real_preview_button, "等待手部原点…")
        self.hamer_status_var.set("实机与 MuJoCo 已到 Home · 等待下一帧有效手部并自动归零")
        self.status_var.set("HAMER REAL WAITING ORIGIN")
        self.log(
            "实体 CR3 和 MuJoCo 已到统一 Home；验证阶段的仿真偏移已丢弃。"
            "下一帧有效 HaMeR 手位将自动成为新的实机控制原点。"
        )

    def _set_hamer_real_home_origin(self, cam_t: np.ndarray) -> None:
        home_ee = self._set_sim_home_now()
        self.hamer_mapper.calibrate(cam_t, home_ee)
        self.hamer_last_valid_hand_at = time.monotonic()
        self.hamer_real_stage = "ready"
        self.hamer_real_confirm_until = (
            time.monotonic() + HAMER_REAL_CONFIRMATION_WINDOW_S
        )
        self._set_button_text(
            self.hamer_real_preview_button,
            "最终确认 HaMeR 实机同步",
        )
        self.hamer_status_var.set("Home 原点已重建 · 30 秒内进行第二次确认")
        self.status_var.set("HAMER REAL READY")
        self.log(
            "HaMeR 实机原点已自动重建："
            f"cam_t={np.round(cam_t, 4).tolist()}，"
            f"Home Link6={np.round(home_ee, 4).tolist()}。"
        )

    def confirm_hamer_real_sync(self) -> None:
        """Second confirmation: reconnect 30003 and begin continuous ServoJ."""
        now = time.monotonic()
        if self.hamer_real_stage != "ready":
            return
        if now > self.hamer_real_confirm_until:
            self.stop_hamer_real_sync("HaMeR 实机同步第二次确认已超时，请重新准备。")
            return
        if self.monitor is None or self.hamer_tracker is None:
            self.stop_hamer_real_sync("HaMeR 实机同步准备失效：反馈或摄像头已断开。")
            return
        try:
            state = self.monitor.require_fresh(max_age=0.5)
            require_motion_ready(state)
            home_deg = sim_rad_to_real_deg(core.HOME_Q_RAD)
            if not joint_target_reached(state, home_deg):
                raise RuntimeError("实体 CR3 已离开 Home 或仍在运动")
            snapshot = self.hamer_tracker.get()
            cam_t = hamer_cam_t(snapshot.result)
            if cam_t is None:
                raise RuntimeError("当前没有有效手部 cam_t")
        except Exception as exc:
            self.log(f"HaMeR 实机同步最终检查失败：{exc}")
            return
        robot_ip = self._active_robot_ip()
        if robot_ip is None:
            return

        if not messagebox.askyesno(
            "HaMeR 实机同步 · 最终确认",
            "实体 CR3 与 MuJoCo 已在统一 Home，当前手位将再次归零。\n\n"
            f"IP: {robot_ip}\n"
            f"实机连续 ServoJ 限速: {HAMER_REAL_HOME_SPEED_DEG_S:.1f} deg/s\n"
            f"手部丢失保护: {HAMER_REAL_HAND_WATCHDOG_S:.1f} s\n\n"
            "确认开始由 HaMeR 控制实体机械臂吗？",
            icon="warning",
        ):
            self.log("用户取消了 HaMeR 实机同步最终确认。")
            return

        # The modal confirmation can remain open for an arbitrary time. Read
        # both robot and hand again after it closes instead of using the values
        # that were checked before the dialog.
        try:
            state = self.monitor.require_fresh(max_age=0.5)
            require_motion_ready(state)
            if not joint_target_reached(state, home_deg):
                raise RuntimeError("最终确认期间实体 CR3 离开了 Home")
            snapshot = self.hamer_tracker.get()
            cam_t = hamer_cam_t(snapshot.result)
            if cam_t is None:
                raise RuntimeError("最终确认后没有有效手部 cam_t")
            hand_age = time.monotonic() - snapshot.result_at
            if snapshot.result_at <= 0.0 or hand_age > HAMER_REAL_HAND_WATCHDOG_S:
                raise RuntimeError(f"最终确认后的手部结果已过期（{hand_age:.2f}s）")
        except Exception as exc:
            self.log(f"HaMeR 实机同步未启动：{exc}")
            return

        # Re-zero once more at the exact handover moment, guaranteeing the
        # first real target is Home even if the hand moved after validation.
        now = time.monotonic()
        home_ee = self._set_sim_home_now()
        self.hamer_mapper.calibrate(cam_t, home_ee)
        self.hamer_last_valid_hand_at = (
            snapshot.result_at if snapshot.result_at > 0.0 else now
        )
        self.hamer_real_stage = "connecting"
        self.hamer_real_confirm_until = 0.0
        self.real_stop_event = threading.Event()
        self._set_button_text(self.hamer_real_preview_button, "停止 HaMeR 实机同步")
        self.hamer_status_var.set("最终确认完成 · 正在连接 30003")
        self.status_var.set("HAMER REAL CONNECTING")
        def work():
            hardware = LiveServoHardware(
                robot_ip,
                max_joint_speed_deg_s=HAMER_REAL_HOME_SPEED_DEG_S,
                feedback=self.monitor,
            )
            try:
                connected_state = hardware.connect()
                if not joint_target_reached(connected_state, home_deg):
                    raise RuntimeError("30003 连接完成时实体 CR3 已不在 Home")
                return hardware, connected_state
            except Exception:
                hardware.close()
                raise

        self._background(work, self._hamer_real_connected, self._hamer_real_failed)

    def _hamer_real_connected(self, result) -> None:
        hardware, state = result
        if self.hamer_real_stage != "connecting" or self.estop_latched:
            hardware.close()
            return
        self.live_hardware = hardware
        self.last_applied_live_speed = hardware.max_joint_speed_deg_s
        self.hamer_real_stage = "active"
        self._set_sim_from_real(state.joints_deg)
        self.q_target = core.HOME_Q_RAD.copy()
        hardware.start_stream(sim_rad_to_real_deg(self.q_target))
        self.hamer_status_var.set("HaMeR 实机同步 ACTIVE · 5°/s ServoJ · 手部丢失自动停止")
        self.status_var.set("HAMER REAL SYNC ACTIVE")
        self.log("HaMeR 实机同步已启动；当前控制链为 HaMeR → MuJoCo IK → CR3 ServoJ。")

    def stop_hamer_real_sync(self, reason: str | None = None) -> None:
        previous_stage = self.hamer_real_stage
        self.real_stop_event.set()
        self.hamer_real_stage = "idle"
        self.hamer_real_confirm_until = 0.0
        hardware = None
        if previous_stage == "active" and self.live_hardware is not None:
            hardware = self.live_hardware
            self.live_hardware = None
            self.last_applied_live_speed = None
        if hasattr(self, "hamer_real_preview_button"):
            self._set_button_text(self.hamer_real_preview_button, "HaMeR 实机同步")
        if hardware is not None:
            self._background(hardware.close, lambda _=None: None, lambda exc: self.log(f"关闭 HaMeR 30003 失败：{exc}"))
        if previous_stage != "idle":
            self.status_var.set("HAMER SIM READY")
        if reason:
            self.log(reason)

    # Compatibility with older call sites while the button name has changed.
    def stop_hamer_real_preview(self, reason: str | None = None) -> None:
        self.stop_hamer_real_sync(reason)

    def _hamer_real_failed(self, exc: Exception) -> None:
        self.stop_hamer_real_sync()
        self.hamer_status_var.set(f"HaMeR 实机同步已中止：{exc}")
        self.status_var.set("HAMER REAL SYNC FAILED")
        self.log(f"HaMeR 实机同步已中止：{exc}")

    def _check_hamer_real_sync(self, now: float) -> None:
        if self.hamer_real_stage == "ready" and now > self.hamer_real_confirm_until:
            self.stop_hamer_real_sync("HaMeR 实机同步第二次确认已超时。")
            return
        if self.hamer_real_stage != "active":
            return
        if now - self.hamer_last_valid_hand_at > HAMER_REAL_HAND_WATCHDOG_S:
            self.stop_hamer_real_sync(
                f"HaMeR 有效手部数据超过 {HAMER_REAL_HAND_WATCHDOG_S:.1f}s 未更新，实机发送已停止。"
            )

    def setup_hamer_origin_video(self) -> None:
        """First press opens the video; second press confirms its current frame."""
        if self.hamer_phase == "origin_video":
            self.confirm_hamer_origin_frame()
            return
        if self.hamer_phase == "origin_starting":
            self.log("原点视频正在打开，请稍候再按第二次。")
            return
        if self.real_busy or self.teach_active or self.teach_starting:
            self.log("真实回放或示教拖拽活动时不能设置 HaMeR 原点。")
            return
        if self.live_hardware is not None or self.live_starting:
            self.log("请先停止实时实机同步；HaMeR 当前仅控制 MuJoCo。")
            return
        if self.hamer_phase in ("live_starting", "live_wait_origin", "live"):
            self.log("请先停止 HaMeR 实时仿真，再重新设置原点。")
            return
        if self.hamer_phase == "origin_validation":
            self.log("原点验证视频正在控制 MuJoCo，请等待播放完成。")
            return
        if self.hamer_starting or self.hamer_tracker is not None:
            self.log("原点设置视频已在处理。")
            return
        try:
            mapper = self._new_hamer_mapper()
            max_ee_step = float(self.hamer_ee_step_var.get())
            video_text = self.hamer_video_var.get().strip()
            if not video_text:
                raise ValueError("原点定义视频路径不能为空")
            tracker = HamerBridgeTracker(
                self.hamer_bridge_url_var.get(),
                video_path=video_text,
            )
        except Exception as exc:
            self.log(f"HaMeR 原点视频设置无效：{exc}")
            return

        self.mode_var.set("hamer")
        self.hamer_mapper = mapper
        self.hamer_max_ee_step = max_ee_step
        self.hamer_phase = "origin_starting"
        self.hamer_last_sequence = 0
        self.hamer_latest_frame = None
        self.hamer_latest_frame_sequence = 0
        self.hamer_rendered_frame_key = None
        if self.hamer_preview_image_id is not None:
            self.canvas.delete(self.hamer_preview_image_id)
            self.hamer_preview_image_id = None
            self.hamer_preview_photo = None
        self.hamer_starting = True
        self._set_button_text(self.hamer_origin_button, "正在打开测试视频…")
        self.hamer_origin_button.configure(state=tk.DISABLED)
        self.hamer_button.configure(state=tk.DISABLED)
        self.hamer_stop_video_button.configure(state=tk.NORMAL)
        self.hamer_status_var.set("Testing HaMeR · 正在打开内置测试视频…")
        self.status_var.set("HAMER ORIGIN VIDEO")
        self.log("原点视频正在打开；画面出现后，第二次点击同一按钮确定当前帧为原点。")

        def work():
            tracker.start()
            return tracker

        self._background(work, self._hamer_started, self._hamer_start_failed)

    def start_hamer(self) -> None:
        """Compatibility alias for the GUI's HaMeR live mode."""
        self.start_hamer_live()

    def start_hamer_live(self) -> None:
        if self.real_busy or self.teach_active or self.teach_starting:
            self.log("真实回放或示教拖拽活动时不能启动 HaMeR 实时仿真。")
            return
        if self.live_hardware is not None or self.live_starting:
            self.log("请先停止实时实机同步；HaMeR 实时模式当前仅控制 MuJoCo。")
            return
        if self.hamer_starting or self.hamer_tracker is not None:
            return
        try:
            mapper = self._new_hamer_mapper()
            max_ee_step = float(self.hamer_ee_step_var.get())
            camera_id = int(self.hamer_camera_id_var.get())
            tracker = HamerBridgeTracker(
                self.hamer_bridge_url_var.get(),
                video_path=None,
                camera_id=camera_id,
            )
        except Exception as exc:
            self.log(f"HaMeR 实时设置无效：{exc}")
            return

        self.mode_var.set("hamer")
        self.hamer_mapper = mapper
        self.hamer_max_ee_step = max_ee_step
        self.hamer_phase = "live_starting"
        self.hamer_last_sequence = 0
        self.hamer_latest_frame = None
        self.hamer_latest_frame_sequence = 0
        self.hamer_rendered_frame_key = None
        self.hamer_starting = True
        self._set_button_text(self.hamer_button, "正在打开摄像头…")
        self.hamer_button.configure(state=tk.DISABLED)
        self.hamer_origin_button.configure(state=tk.DISABLED)
        self.hamer_stop_video_button.configure(state=tk.DISABLED)
        self.hamer_status_var.set("HaMeR 正在启动实时摄像头 · 启动后按 R 设定原点")
        self.status_var.set("STARTING HAMER LIVE SIM")

        def work():
            tracker.start()
            return tracker

        self._background(work, self._hamer_started, self._hamer_start_failed)

    def _hamer_started(self, tracker: HamerBridgeTracker) -> None:
        if not self.hamer_starting or self.mode_var.get() != "hamer":
            tracker.stop()
            return
        self.hamer_starting = False
        self.hamer_tracker = tracker
        if self.hamer_phase == "origin_starting":
            self.hamer_phase = "origin_video"
            self.hamer_origin_button.configure(
                text="确定测试帧",
                state=tk.NORMAL,
            )
            self.hamer_button.configure(state=tk.DISABLED)
            self.hamer_stop_video_button.configure(state=tk.NORMAL)
            self.hamer_status_var.set("Testing HaMeR · 选择合适画面后再点击同一按钮开始验证")
            self.status_var.set("HAMER ORIGIN PREVIEW")
        elif self.hamer_phase == "live_starting":
            self.hamer_phase = "live_wait_origin"
            self._set_button_text(self.hamer_button, "停止 HaMeR 实时")
            self.hamer_button.configure(state=tk.NORMAL)
            self.hamer_origin_button.configure(state=tk.DISABLED)
            self.hamer_stop_video_button.configure(state=tk.DISABLED)
            self.hamer_status_var.set("HaMeR 实时摄像头已启动 · 请按 R 设定当前手位为原点")
            self.status_var.set("HAMER LIVE WAITING FOR R")
            self.log("HaMeR 实时摄像头已启动；按 R 之前只显示画面，不移动 MuJoCo。")
        else:
            tracker.stop()
            self.hamer_tracker = None

    def _hamer_start_failed(self, exc: Exception) -> None:
        failed_phase = self.hamer_phase
        self.hamer_starting = False
        self.hamer_tracker = None
        if failed_phase in ("origin_starting", "live_starting"):
            self.hamer_mapper.clear_origin()
        self.hamer_phase = "idle"
        self._set_button_text(self.hamer_button, "HaMeR 实时仿真")
        self.hamer_button.configure(state=tk.NORMAL)
        self._set_button_text(self.hamer_origin_button, "Testing HaMeR")
        self.hamer_origin_button.configure(state=tk.NORMAL)
        self.hamer_stop_video_button.configure(state=tk.DISABLED)
        self.hamer_status_var.set(f"HaMeR 启动失败：{exc}")
        self.status_var.set("HAMER START FAILED")
        self.log(f"HaMeR 启动失败：{exc}")

    def stop_hamer(self, *, clear_origin: bool = False) -> None:
        self.stop_hamer_real_preview()
        previous_phase = self.hamer_phase
        self.hamer_starting = False
        tracker = self.hamer_tracker
        self.hamer_tracker = None
        if clear_origin:
            self.hamer_mapper.clear_origin()
        self.hamer_phase = "idle"
        self.hamer_latest_frame = None
        self.hamer_latest_frame_sequence = 0
        self.hamer_rendered_frame_key = None
        self.hamer_last_sequence = 0
        self._set_button_text(self.hamer_button, "HaMeR 实时仿真")
        self.hamer_button.configure(state=tk.NORMAL)
        self._set_button_text(self.hamer_origin_button, "Testing HaMeR")
        self.hamer_origin_button.configure(state=tk.NORMAL)
        self.hamer_stop_video_button.configure(state=tk.DISABLED)
        self.hamer_status_var.set("Testing HaMeR=测试视频 · 实时摄像头中按 R 设定原点")
        if self.hamer_preview_image_id is not None:
            self.canvas.delete(self.hamer_preview_image_id)
            self.hamer_preview_image_id = None
            self.hamer_preview_photo = None
        if tracker is not None:
            self._background(tracker.stop, lambda _=None: None, lambda exc: self.log(f"HaMeR 停止异常：{exc}"))
        if previous_phase in ("live", "live_wait_origin", "live_starting"):
            self.status_var.set("HAMER LIVE STOPPED")
            self.log("HaMeR 实时仿真已停止。")

    def stop_hamer_test_video(self) -> None:
        if self.hamer_phase not in ("origin_starting", "origin_video", "origin_validation"):
            self.log("当前没有正在播放的 Testing HaMeR 视频。")
            return
        self.stop_hamer(clear_origin=True)
        self.status_var.set("HAMER TEST VIDEO STOPPED")
        self.hamer_status_var.set("Testing HaMeR 视频已停止 · 可重新测试或启动实时摄像头")
        self.log("Testing HaMeR 视频已手动停止；MuJoCo 保持当前位姿。")

    def set_hamer_origin(self) -> None:
        """Compatibility alias for setting the live-camera origin."""
        self.set_hamer_camera_origin()

    def set_hamer_camera_origin(self) -> None:
        """Bind the current live-camera hand pose to the current MuJoCo Link6."""
        tracker = self.hamer_tracker
        if tracker is None or self.hamer_phase not in ("live_wait_origin", "live"):
            self.log("请先启动“HaMeR 实时仿真”，然后在摄像头画面中按 R 设定原点。")
            return
        snapshot = tracker.get()
        cam_t = hamer_cam_t(snapshot.result)
        if cam_t is None:
            latency = (
                f"{snapshot.request_latency_s:.1f}s"
                if snapshot.request_latency_s > 0.0
                else "尚无返回"
            )
            self.log(
                "实时摄像头当前没有有效手部 cam_t；"
                f"桥接状态：{snapshot.status}；最近推理耗时：{latency}。"
            )
            self.hamer_status_var.set(
                f"尚未取得 cam_t · {snapshot.status} · 推理 {latency}"
            )
            return
        current_q = self.data.qpos[self.arm_qpos_indices].copy()
        self.q_target = current_q
        ee_pos = self.data.xpos[self.end_effector_id].copy()
        self.hamer_mapper.calibrate(cam_t, ee_pos)
        self.hamer_phase = "live"
        self.hamer_last_sequence = snapshot.sequence
        self.hamer_status_var.set("HaMeR LIVE · 摄像头原点已设定 · 按 R 可随时重新归零")
        self.status_var.set("HAMER LIVE TRACKING")
        self.log(
            "HaMeR 实时摄像头原点已设定："
            f"cam_t={np.round(cam_t, 4).tolist()}  Link6={np.round(ee_pos, 4).tolist()}"
        )

    def confirm_hamer_origin_frame(self) -> None:
        """Use the currently displayed origin-video frame as the origin."""
        tracker = self.hamer_tracker
        if tracker is None or self.hamer_phase != "origin_video":
            self.log("请先点击“原点设置视频”打开视频。")
            return
        snapshot = tracker.get()
        cam_t = hamer_cam_t(snapshot.result)
        if cam_t is None:
            latency = (
                f"{snapshot.request_latency_s:.1f}s"
                if snapshot.request_latency_s > 0.0
                else "尚无返回"
            )
            self.log(
                "当前视频帧没有有效手部 cam_t；"
                f"桥接状态：{snapshot.status}；最近推理耗时：{latency}。"
            )
            self.hamer_status_var.set(
                f"当前帧无 cam_t · {snapshot.status} · 推理 {latency}"
            )
            return
        current_q = self.data.qpos[self.arm_qpos_indices].copy()
        self.q_target = current_q
        ee_pos = self.data.xpos[self.end_effector_id].copy()
        self.hamer_mapper.calibrate(cam_t, ee_pos)
        self.hamer_phase = "origin_validation"
        self.hamer_last_sequence = snapshot.sequence
        self._set_button_text(self.hamer_button, "HaMeR 实时仿真")
        self.hamer_button.configure(state=tk.DISABLED)
        self._set_button_text(self.hamer_origin_button, "原点已确定 · 验证中")
        self.hamer_origin_button.configure(state=tk.DISABLED)
        self.hamer_status_var.set("HaMeR 原点已确定 · 原点视频继续控制 MuJoCo 验证映射")
        self.status_var.set("HAMER ORIGIN VIDEO VALIDATION")
        self.log(
            "HaMeR 原点已由第二次按键时的视频帧确定："
            f"cam_t={np.round(cam_t, 4).tolist()}  Link6={np.round(ee_pos, 4).tolist()}"
        )
        self.log("原点视频将继续播放；后续手部位移现在会直接驱动 MuJoCo 机械臂。")

    def _update_hamer(self, now: float) -> None:
        tracker = self.hamer_tracker
        if tracker is None or self.mode_var.get() != "hamer" or now < self.hamer_next_poll:
            return
        self.hamer_next_poll = now + 1.0 / 30.0
        snapshot = tracker.get()
        self.hamer_latest_frame = snapshot.frame_bgr
        self.hamer_latest_frame_sequence = snapshot.frame_sequence
        if self.manual_home_active:
            self.hamer_last_sequence = snapshot.sequence
            return
        if snapshot.status != self.hamer_last_status:
            self.hamer_last_status = snapshot.status
        if snapshot.sequence == self.hamer_last_sequence:
            if not tracker.running and "ended" in snapshot.status.lower():
                if self.hamer_phase == "origin_video":
                    self.hamer_tracker = None
                    self.hamer_phase = "idle"
                    self.hamer_mapper.clear_origin()
                    self._set_button_text(self.hamer_button, "HaMeR 实时仿真")
                    self.hamer_button.configure(state=tk.NORMAL)
                    self._set_button_text(self.hamer_origin_button, "Testing HaMeR")
                    self.hamer_origin_button.configure(state=tk.NORMAL)
                    self.hamer_stop_video_button.configure(state=tk.DISABLED)
                    self.hamer_status_var.set("Testing HaMeR 视频已结束，但尚未选择测试原点 · 可重试")
                elif self.hamer_phase == "origin_validation":
                    self.hamer_tracker = None
                    self.hamer_phase = "idle"
                    self.hamer_mapper.clear_origin()
                    self._set_button_text(self.hamer_button, "HaMeR 实时仿真")
                    self.hamer_button.configure(state=tk.NORMAL)
                    self._set_button_text(self.hamer_origin_button, "Testing HaMeR")
                    self.hamer_origin_button.configure(state=tk.NORMAL)
                    self.hamer_stop_video_button.configure(state=tk.DISABLED)
                    self.hamer_status_var.set("Testing HaMeR 验证视频已播放完 · 可启动实时摄像头")
                    self.status_var.set("HAMER TEST COMPLETE")
                    self.log("Testing HaMeR 验证已完成；MuJoCo 保持最终位姿。")
                else:
                    self.hamer_status_var.set("HaMeR 实时画面已停止 · MuJoCo 保持当前位姿")
                if self.hamer_tracker is None:
                    self._background(
                        tracker.stop,
                        lambda _=None: None,
                        lambda exc: self.log(f"HaMeR 视频释放异常：{exc}"),
                    )
            return
        self.hamer_last_sequence = snapshot.sequence
        cam_t = hamer_cam_t(snapshot.result)
        if cam_t is None:
            if self.hamer_real_stage == "homing":
                self.hamer_status_var.set("HaMeR 实机准备 · 仿真与实机正在回统一 Home")
            elif self.hamer_real_stage == "waiting_origin":
                self.hamer_status_var.set("实机与 MuJoCo 已到 Home · 等待有效手部并自动归零")
            elif self.hamer_real_stage in ("ready", "connecting"):
                self.hamer_status_var.set("Home 已建立 · 当前手部暂时丢失，尚未发送运动")
            elif self.hamer_phase == "origin_video":
                self.hamer_status_var.set("原点视频预览中 · 当前帧未检测到手部")
            elif self.hamer_phase == "origin_validation":
                self.hamer_status_var.set("原点视频验证中 · 当前帧未检测到有效手部")
            elif self.hamer_phase == "live_wait_origin":
                self.hamer_status_var.set("HaMeR 实时摄像头已启动 · 等待检测手部后按 R 设定原点")
            else:
                self.hamer_status_var.set(f"HaMeR LIVE · 未检测到有效手部 · {snapshot.status}")
            return
        self.hamer_last_valid_hand_at = (
            snapshot.result_at if snapshot.result_at > 0.0 else now
        )
        if (
            self.hamer_real_stage == "waiting_origin"
            and snapshot.sequence > self.hamer_real_origin_after_sequence
        ):
            self._set_hamer_real_home_origin(cam_t)
            return
        if self.hamer_real_stage == "waiting_origin":
            return
        if self.hamer_real_stage in ("homing", "ready", "connecting"):
            # Hold both sides exactly at Home until the final confirmation.
            self.q_target = core.HOME_Q_RAD.copy()
            return
        if self.hamer_phase == "origin_video":
            self.hamer_status_var.set("原点视频已检测到手部 · 再点击同一按钮确定当前帧为原点")
            return
        if self.hamer_phase == "live_wait_origin":
            self.hamer_status_var.set("HaMeR 实时摄像头已检测到手部 · 按 R 设定当前手位为原点")
            return
        if self.hamer_phase not in ("live", "origin_validation") or not self.hamer_mapper.calibrated:
            return
        target_pos = self.hamer_mapper.target_pos(cam_t)
        ee_pos = self.data.xpos[self.end_effector_id].copy()
        position_error = target_pos - ee_pos
        error_norm = float(np.linalg.norm(position_error))
        if error_norm > self.hamer_max_ee_step:
            position_error *= self.hamer_max_ee_step / error_norm
        twist = np.zeros(6)
        twist[:3] = position_error
        current_q = self.data.qpos[self.arm_qpos_indices].copy()
        self.q_target = core.apply_cartesian_increment(
            self.model,
            self.data,
            self.end_effector_id,
            self.arm_dof_indices,
            self.arm_joint_ids,
            current_q,
            twist,
        )
        delta = target_pos - self.hamer_mapper.ee_origin
        source_text = "原点视频验证" if self.hamer_phase == "origin_validation" else "HaMeR LIVE"
        control_text = (
            "实体 CR3 同步 ACTIVE"
            if self.hamer_real_stage == "active"
            else "仅 MuJoCo"
        )
        self.hamer_status_var.set(
            f"{source_text} · {control_text} · "
            f"ΔXYZ={np.round(delta * 1000.0, 1).tolist()} mm"
        )

    def open_quest_settings(self) -> None:
        if self.quest_receiver is not None:
            self.log("请先停止 Quest 接收，再修改网络设置。")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title(self._tr("Quest 设置", "Quest Settings"))
        dialog.configure(bg="#111c2e")
        dialog.resizable(False, False)
        frame = ttk.Frame(dialog, padding=16, style="Card.TFrame")
        frame.pack(fill=tk.BOTH, expand=True)

        protocol_var = tk.StringVar(value=self.quest_protocol_var.get())
        host_var = tk.StringVar(value=self.quest_host_var.get())
        port_var = tk.StringVar(value=self.quest_port_var.get())
        hand_var = tk.StringVar(value=self.quest_hand_var.get())
        gain_var = tk.StringVar(value=self.quest_gain_var.get())
        deadzone_mm_var = tk.StringVar(value=self.quest_deadzone_mm_var.get())
        slow_alpha_var = tk.StringVar(value=self.quest_slow_alpha_var.get())
        fast_alpha_var = tk.StringVar(value=self.quest_fast_alpha_var.get())
        cartesian_speed_var = tk.StringVar(
            value=self.quest_cartesian_speed_var.get()
        )
        rows = (
            (self._tr("协议", "Protocol"), protocol_var, ("udp", "tcp")),
            (self._tr("监听地址", "Listen Host"), host_var, None),
            (self._tr("端口", "Port"), port_var, None),
            (self._tr("控制手", "Control Hand"), hand_var, ("right", "left")),
            (self._tr("位移增益", "Motion Gain"), gain_var, None),
            (self._tr("静止死区 (mm)", "Idle Deadzone (mm)"), deadzone_mm_var, None),
            (self._tr("慢速滤波 Alpha", "Slow Filter Alpha"), slow_alpha_var, None),
            (self._tr("快速滤波 Alpha", "Fast Filter Alpha"), fast_alpha_var, None),
            (self._tr("笛卡尔追踪速度 (m/s)", "Cartesian Tracking (m/s)"), cartesian_speed_var, None),
        )
        for row, (label, variable, choices) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            if choices is None:
                widget = ttk.Entry(frame, textvariable=variable, width=22)
            else:
                widget = ttk.Combobox(
                    frame,
                    textvariable=variable,
                    values=choices,
                    state="readonly",
                    width=19,
                )
            widget.grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=4)

        ttk.Label(
            frame,
            text=self._tr(
                "UDP 默认监听 0.0.0.0:9000；Quest 内填这台电脑的局域网 IP。\n"
                "USB TCP 需要先执行 adb reverse tcp:8000 tcp:8000。",
                "UDP normally listens on 0.0.0.0:9000; enter this PC's LAN IP in Quest.\n"
                "USB TCP requires adb reverse tcp:8000 tcp:8000 first.",
            ),
            wraplength=390,
            justify=tk.LEFT,
        ).grid(row=len(rows), column=0, columnspan=2, sticky="w", pady=(10, 6))

        def save() -> None:
            try:
                protocol = protocol_var.get().strip().lower()
                if protocol not in {"udp", "tcp"}:
                    raise ValueError("协议必须是 udp 或 tcp")
                port = int(port_var.get())
                if not 1 <= port <= 65535:
                    raise ValueError("端口必须在 1 到 65535 之间")
                hand = hand_var.get().strip().lower()
                if hand not in {"right", "left"}:
                    raise ValueError("控制手必须是 right 或 left")
                gain = float(gain_var.get())
                if not np.isfinite(gain) or not 0.1 <= gain <= 3.0:
                    raise ValueError("位移增益必须在 0.1 到 3.0 之间")
                deadzone_mm = float(deadzone_mm_var.get())
                if not np.isfinite(deadzone_mm) or not 0.0 <= deadzone_mm <= 10.0:
                    raise ValueError("静止死区必须在 0 到 10 mm 之间")
                slow_alpha = float(slow_alpha_var.get())
                fast_alpha = float(fast_alpha_var.get())
                if not 0.05 <= slow_alpha <= fast_alpha <= 1.0:
                    raise ValueError("滤波 Alpha 必须满足 0.05 ≤ 慢速 ≤ 快速 ≤ 1.0")
                cartesian_speed = float(cartesian_speed_var.get())
                if not np.isfinite(cartesian_speed) or not 0.05 <= cartesian_speed <= 1.5:
                    raise ValueError("笛卡尔追踪速度必须在 0.05 到 1.5 m/s 之间")
                host = host_var.get().strip() or "0.0.0.0"
            except ValueError as exc:
                messagebox.showerror(
                    self._tr("参数无效", "Invalid Settings"),
                    self._tr(str(exc)),
                    parent=dialog,
                )
                return
            self.quest_protocol_var.set(protocol)
            self.quest_host_var.set(host)
            self.quest_port_var.set(str(port))
            self.quest_hand_var.set(hand)
            self.quest_gain_var.set(f"{gain:g}")
            self.quest_deadzone_mm_var.set(f"{deadzone_mm:g}")
            self.quest_slow_alpha_var.set(f"{slow_alpha:g}")
            self.quest_fast_alpha_var.set(f"{fast_alpha:g}")
            self.quest_cartesian_speed_var.set(f"{cartesian_speed:g}")
            self.quest_mapper.gain = gain
            dialog.destroy()

        ttk.Button(frame, text=self._tr("保存", "Save"), command=save).grid(
            row=len(rows) + 1, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        dialog.transient(self.root)
        dialog.grab_set()

    def toggle_quest(self) -> None:
        if self.quest_receiver is None:
            self.start_quest()
        else:
            self.stop_quest()

    def start_quest(self) -> None:
        if (
            self.real_busy
            or self.teach_active
            or self.live_hardware is not None
            or self.hamer_real_stage != "idle"
        ):
            self.log("实机回放、实时同步或示教活动时不能启动 Quest 仿真。")
            return
        if self.hamer_tracker is not None or self.hamer_starting:
            self.stop_hamer()
        self.mode_var.set("quest")
        self.on_mode_changed()
        try:
            port = int(self.quest_port_var.get())
            gain = float(self.quest_gain_var.get())
            deadzone_m = float(self.quest_deadzone_mm_var.get()) / 1000.0
            slow_alpha = float(self.quest_slow_alpha_var.get())
            fast_alpha = float(self.quest_fast_alpha_var.get())
            cartesian_speed = float(self.quest_cartesian_speed_var.get())
            if not np.isfinite(gain) or not 0.1 <= gain <= 3.0:
                raise ValueError("Quest motion gain must be in 0.1..3.0")
            if not 0.0 <= deadzone_m <= 0.010:
                raise ValueError("Quest deadzone must be in 0..10 mm")
            if not 0.05 <= slow_alpha <= fast_alpha <= 1.0:
                raise ValueError("Quest filter alpha must satisfy 0.05 <= slow <= fast <= 1")
            if not 0.05 <= cartesian_speed <= 1.5:
                raise ValueError("Quest Cartesian speed must be in 0.05..1.5 m/s")
            mapper = QuestWristMapper(
                gain=gain,
                deadzone_m=deadzone_m,
                ema_alpha=slow_alpha,
                fast_ema_alpha=fast_alpha,
                filter_reference_hz=QUEST_CONTROL_RATE_HZ,
                max_predict_s=QUEST_MAX_PREDICT_S,
            )
            receiver = QuestHandReceiver(
                protocol=self.quest_protocol_var.get(),
                host=self.quest_host_var.get(),
                port=port,
            )
            receiver.start()
        except (OSError, ValueError) as exc:
            self.quest_status_var.set(f"Quest 接收启动失败：{exc}")
            self.log(f"Quest 接收启动失败：{exc}")
            return
        self.quest_mapper = mapper
        self.quest_receiver = receiver
        self.quest_phase = "wait_origin"
        self.quest_last_sequence = 0
        self.quest_last_control_time = 0.0
        self.quest_last_wrist_received_at = 0.0
        self._set_button_text(self.quest_button, "停止 Quest 接收")
        self.quest_status_var.set(
            f"Quest {receiver.protocol.upper()} {receiver.host}:{receiver.port} · "
            f"等待 {self.quest_hand_var.get()} 腕部数据"
        )
        self.status_var.set("QUEST WAITING FOR WRIST")
        self.log(
            f"Quest {receiver.protocol.upper()} 正在监听 "
            f"{receiver.host}:{receiver.port}，控制手={self.quest_hand_var.get()}。"
        )
        self.log("Quest 实时接收已启动；按 R 前只显示腕部数据，不移动 MuJoCo。")

    def stop_quest(self, *, clear_origin: bool = True) -> None:
        self.stop_quest_real_sync()
        receiver = self.quest_receiver
        self.quest_receiver = None
        if receiver is not None:
            receiver.stop()
        if clear_origin:
            self.quest_mapper.clear_origin()
            self.quest_orientation_target = None
        self.quest_phase = "idle"
        self.quest_last_sequence = 0
        self.quest_last_control_time = 0.0
        self._set_button_text(self.quest_button, "启动 Quest 接收")
        self.quest_status_var.set("Quest 未启动 · 仅控制 MuJoCo")
        if receiver is not None:
            self.log("Quest 接收已停止。")

    def set_quest_origin(self) -> None:
        receiver = self.quest_receiver
        if receiver is None:
            self.log("请先启动 Quest 接收。")
            return
        snapshot = receiver.get(self.quest_hand_var.get())
        if (
            not snapshot.has_wrist
            or time.monotonic() - snapshot.received_at > 0.75
        ):
            self.log("Quest 当前没有新鲜腕部数据；请在头显中启动 Streaming。")
            return
        current_q = self.data.qpos[self.arm_qpos_indices].copy()
        self.q_target = current_q
        mujoco.mj_forward(self.model, self.data)
        ee_pos = self.data.xpos[self.end_effector_id].copy()
        self.quest_orientation_target = self.data.xmat[
            self.end_effector_id
        ].reshape(3, 3).copy()
        assert snapshot.wrist_position is not None
        self.quest_mapper.calibrate(
            snapshot.wrist_position,
            ee_pos,
            timestamp=snapshot.received_at,
        )
        self.quest_phase = "live"
        self.quest_last_sequence = snapshot.wrist_sequence
        self.quest_last_control_time = time.monotonic()
        self.quest_last_wrist_received_at = snapshot.received_at
        self.status_var.set("QUEST SIM ACTIVE")
        self.quest_status_var.set(
            f"QUEST LIVE · {snapshot.side} 腕部原点已设定 · 仅 MuJoCo"
        )
        self.log(
            "Quest 腕部原点已设定："
            f"wrist={np.round(snapshot.wrist_position, 4).tolist()}  "
            f"Link6={np.round(ee_pos, 4).tolist()}"
        )

    def _update_quest(self, now: float) -> None:
        receiver = self.quest_receiver
        if (
            self.manual_home_active
            or receiver is None
            or self.mode_var.get() != "quest"
            or now < self.quest_next_poll
        ):
            return
        quest_period = 1.0 / QUEST_CONTROL_RATE_HZ
        if (
            self.quest_next_poll <= 0.0
            or now - self.quest_next_poll > quest_period
        ):
            self.quest_next_poll = now + quest_period
        else:
            # Advance from the previous deadline. Advancing from ``now``
            # quantizes 60 Hz down to 50 Hz on the GUI's 10 ms timer.
            self.quest_next_poll += quest_period
        snapshot = receiver.get(self.quest_hand_var.get())
        if not snapshot.has_wrist or now - snapshot.received_at > 0.75:
            self.quest_status_var.set(
                f"Quest {receiver.protocol.upper()} · 等待 "
                f"{self.quest_hand_var.get()} 腕部数据"
            )
            return
        self.quest_last_sequence = snapshot.wrist_sequence
        self.quest_last_wrist_received_at = snapshot.received_at
        assert snapshot.wrist_position is not None
        self.quest_last_valid_wrist_at = snapshot.received_at
        if (
            self.quest_real_stage == "waiting_origin"
            and snapshot.wrist_sequence > self.quest_real_origin_after_sequence
        ):
            self._set_quest_real_home_origin(
                snapshot.wrist_position,
                snapshot.received_at,
            )
            return
        if self.quest_real_stage == "waiting_origin":
            return
        if self.quest_real_stage in ("homing", "ready", "connecting"):
            self.q_target = core.HOME_Q_RAD.copy()
            return
        wrist_mm = np.round(snapshot.wrist_position * 1000.0, 1).tolist()
        landmark_count = 0 if snapshot.landmarks is None else len(snapshot.landmarks)
        if self.quest_phase == "wait_origin" or not self.quest_mapper.calibrated:
            self.quest_status_var.set(
                f"Quest {snapshot.side} 已跟踪 · XYZ={wrist_mm} mm · "
                f"landmarks={landmark_count} · 按 R 设原点"
            )
            return
        target_pos = self.quest_mapper.target_pos(
            snapshot.wrist_position,
            timestamp=snapshot.received_at,
            now=now,
        )
        ee_pos = self.data.xpos[self.end_effector_id].copy()
        position_error = target_pos - ee_pos
        # Use the actual GUI control interval, capped for safety.  The old
        # fixed 1/60 s assumption made the Cartesian limiter inconsistent with
        # Tk scheduling and could turn occasional delayed callbacks into
        # visible step changes.
        if self.quest_last_control_time <= 0.0:
            control_dt = 1.0 / QUEST_CONTROL_RATE_HZ
        else:
            control_dt = min(
                max(now - self.quest_last_control_time, 1.0e-4),
                QUEST_MAX_CONTROL_DT_S,
            )
        self.quest_last_control_time = now
        position_error = limit_cartesian_tracking_step(
            position_error,
            max_speed_m_s=float(self.quest_cartesian_speed_var.get()),
            dt=control_dt,
        )
        if self.quest_orientation_target is None:
            self.quest_orientation_target = self.data.xmat[
                self.end_effector_id
            ].reshape(3, 3).copy()
        self.q_target = core.apply_position_increment_holding_orientation(
            self.model,
            self.data,
            self.end_effector_id,
            self.arm_dof_indices,
            self.arm_joint_ids,
            self.q_target,
            position_error,
            self.quest_orientation_target,
        )
        assert self.quest_mapper.ee_origin is not None
        delta_mm = np.round(
            (target_pos - self.quest_mapper.ee_origin) * 1000.0, 1
        ).tolist()
        self.quest_status_var.set(
            f"QUEST LIVE · ΔXYZ={delta_mm} mm · "
            f"{self.quest_mapper.last_sample_hz:.0f} Hz · "
            f"age={(now - snapshot.received_at) * 1000.0:.0f} ms · "
            f"v={self.quest_mapper.last_wrist_speed_m_s:.2f} m/s · "
            f"α={self.quest_mapper.last_alpha:.2f} · "
            f"landmarks={landmark_count} · packets={snapshot.packets_received}"
        )

    def toggle_quest_real_sync(self) -> None:
        if self.quest_real_stage == "idle":
            self.prepare_quest_real_home()
        elif self.quest_real_stage == "ready":
            self.confirm_quest_real_sync()
        elif self.quest_real_stage in ("homing", "waiting_origin", "connecting"):
            self.stop_quest_real_sync("用户取消了 Quest 实机接管流程。")
        elif self.quest_real_stage == "active":
            self.stop_quest_real_sync("用户停止了 Quest 实机同步。")

    def prepare_quest_real_home(self) -> None:
        """First confirmation: align, then move CR3 and MuJoCo to Home."""
        if not self.args.enable_real_execution:
            self.log("Quest 实机同步锁定：请使用 --enable-real-execution 启动 GUI。")
            return
        if self.estop_latched or self.real_busy or self.teach_active:
            self.log("软件急停、真实回放或示教拖拽活动时不能启动 Quest 实机同步。")
            return
        if self.hamer_real_stage != "idle":
            self.log("另一个手部实机接管流程正在运行。")
            return
        receiver = self.quest_receiver
        if (
            receiver is None
            or self.quest_phase != "live"
            or not self.quest_mapper.calibrated
        ):
            self.log("请先启动 Quest 接收，按 R 设定原点，并完成 MuJoCo 仿真验证。")
            return
        if self.live_hardware is not None or self.live_starting:
            self.log("另一个实时实机连接已经活动，请先停止。")
            return
        if self.monitor is None:
            self.log("正在先连接 30004 只读反馈；连接后请再次点击 Quest 实机同步。")
            self.connect_feedback()
            return
        robot_ip = self._active_robot_ip()
        if robot_ip is None:
            return
        try:
            state = self.monitor.require_fresh(max_age=0.5)
            require_motion_ready(state)
            snapshot = receiver.get(self.quest_hand_var.get())
            if (
                not snapshot.has_wrist
                or time.monotonic() - snapshot.received_at > 0.75
            ):
                raise RuntimeError("Quest 腕部数据不新鲜")
        except Exception as exc:
            self.log(f"Quest 实机接管前状态检查失败：{exc}")
            return
        try:
            quest_speed = self._quest_real_speed_deg_s()
        except ValueError as exc:
            self.log(f"参数无效：{exc}")
            return

        home_deg = sim_rad_to_real_deg(core.HOME_Q_RAD)
        if not messagebox.askyesno(
            "Quest 实机同步 · 第一次确认",
            "将冻结 Quest 仿真目标，并让实体 CR3 与 MuJoCo 自动回到统一 Home。\n\n"
            f"当前实机关节: {np.round(state.joints_deg, 2).tolist()}°\n"
            f"目标 Home: {np.round(home_deg, 2).tolist()}°\n"
            f"回 Home 限速: {quest_speed:.2f} deg/s\n\n"
            "确认工作区安全并开始低速回 Home 吗？",
            icon="warning",
        ):
            self.log("用户取消了 Quest 实机同步第一次确认。")
            return

        self._set_sim_from_real(state.joints_deg)
        self.real_stop_event = threading.Event()
        self.quest_real_stage = "homing"
        self.quest_real_confirm_until = 0.0
        self.q_target = core.HOME_Q_RAD.copy()
        self._set_button_text(self.quest_real_button, "停止回 Home")
        self.quest_status_var.set("Quest 实机准备 · 仿真与实机正在回统一 Home")
        self.status_var.set("QUEST REAL HOMING")
        self.log(
            "Quest 第一次确认完成：已按 30004 自动对齐，"
            f"正在以 {quest_speed:.2f}°/s 回统一 Home。"
        )
        def work():
            hardware = LiveServoHardware(
                robot_ip,
                max_joint_speed_deg_s=quest_speed,
                feedback=self.monitor,
            )
            try:
                hardware.connect()
                return stream_live_target_until_reached(
                    hardware,
                    home_deg,
                    stop_event=self.real_stop_event,
                )
            finally:
                hardware.close()

        self._background(
            work,
            self._quest_real_home_complete,
            self._quest_real_failed,
        )

    def _quest_real_home_complete(self, _state) -> None:
        if self.quest_real_stage != "homing" or self.estop_latched:
            return
        self._set_sim_home_now()
        self.quest_real_stage = "waiting_origin"
        receiver = self.quest_receiver
        self.quest_real_origin_after_sequence = (
            0
            if receiver is None
            else receiver.get(self.quest_hand_var.get()).wrist_sequence
        )
        self._set_button_text(self.quest_real_button, "等待手部原点…")
        self.quest_status_var.set("实机与 MuJoCo 已到 Home · 等待下一帧 Quest 腕部数据")
        self.status_var.set("QUEST REAL WAITING ORIGIN")
        self.log(
            "实体 CR3 和 MuJoCo 已到统一 Home；仿真验证偏移已丢弃。"
            "下一帧有效 Quest 腕部位置将自动成为新的实机控制原点。"
        )

    def _set_quest_real_home_origin(
        self,
        wrist_position: np.ndarray,
        received_at: float,
    ) -> None:
        home_ee = self._set_sim_home_now()
        self.quest_orientation_target = self.data.xmat[
            self.end_effector_id
        ].reshape(3, 3).copy()
        self.quest_mapper.calibrate(
            wrist_position,
            home_ee,
            timestamp=received_at,
        )
        self.quest_last_wrist_received_at = received_at
        self.quest_last_valid_wrist_at = received_at
        self.quest_real_stage = "ready"
        self.quest_real_confirm_until = (
            time.monotonic() + HAMER_REAL_CONFIRMATION_WINDOW_S
        )
        self._set_button_text(
            self.quest_real_button,
            "最终确认 Quest 实机同步",
        )
        self.quest_status_var.set("Quest Home 原点已重建 · 30 秒内进行第二次确认")
        self.status_var.set("QUEST REAL READY")
        self.log(
            "Quest 实机原点已自动重建："
            f"wrist={np.round(wrist_position, 4).tolist()}，"
            f"Home Link6={np.round(home_ee, 4).tolist()}。"
        )

    def confirm_quest_real_sync(self) -> None:
        """Second confirmation: open 30003 and start Quest ServoJ control."""
        now = time.monotonic()
        if self.quest_real_stage != "ready":
            return
        if now > self.quest_real_confirm_until:
            self.stop_quest_real_sync("Quest 实机同步第二次确认已超时。")
            return
        receiver = self.quest_receiver
        if self.monitor is None or receiver is None:
            self.stop_quest_real_sync("Quest 实机同步准备失效：反馈或 Quest 接收已断开。")
            return
        try:
            state = self.monitor.require_fresh(max_age=0.5)
            require_motion_ready(state)
            home_deg = sim_rad_to_real_deg(core.HOME_Q_RAD)
            if not joint_target_reached(state, home_deg):
                raise RuntimeError("实体 CR3 已离开 Home 或仍在运动")
            snapshot = receiver.get(self.quest_hand_var.get())
            if (
                not snapshot.has_wrist
                or now - snapshot.received_at > 0.75
            ):
                raise RuntimeError("当前没有新鲜 Quest 腕部数据")
        except Exception as exc:
            self.log(f"Quest 实机同步最终检查失败：{exc}")
            return
        try:
            quest_speed = self._quest_real_speed_deg_s()
        except ValueError as exc:
            self.log(f"参数无效：{exc}")
            return
        robot_ip = self._active_robot_ip()
        if robot_ip is None:
            return

        if not messagebox.askyesno(
            "Quest 实机同步 · 最终确认",
            "实体 CR3 与 MuJoCo 已在统一 Home，当前 Quest 腕部将再次归零。\n\n"
            f"IP: {robot_ip}\n"
            f"实机连续 ServoJ 限速: {quest_speed:.2f} deg/s\n"
            f"腕部数据丢失保护: {HAMER_REAL_HAND_WATCHDOG_S:.1f} s\n\n"
            "确认开始由 Quest XYZ 控制实体机器臂吗？",
            icon="warning",
        ):
            self.log("用户取消了 Quest 实机同步最终确认。")
            return

        try:
            now = time.monotonic()
            state = self.monitor.require_fresh(max_age=0.5)
            require_motion_ready(state)
            if not joint_target_reached(state, home_deg):
                raise RuntimeError("最终确认期间实体 CR3 离开了 Home")
            snapshot = receiver.get(self.quest_hand_var.get())
            if not snapshot.has_wrist:
                raise RuntimeError("最终确认后没有 Quest 腕部数据")
            wrist_age = now - snapshot.received_at
            if wrist_age > 0.75:
                raise RuntimeError(f"最终确认后的 Quest 腕部数据已过期（{wrist_age:.2f}s）")
        except Exception as exc:
            self.log(f"Quest 实机同步未启动：{exc}")
            return

        assert snapshot.wrist_position is not None
        home_ee = self._set_sim_home_now()
        self.quest_orientation_target = self.data.xmat[
            self.end_effector_id
        ].reshape(3, 3).copy()
        self.quest_mapper.calibrate(
            snapshot.wrist_position,
            home_ee,
            timestamp=snapshot.received_at,
        )
        self.quest_last_wrist_received_at = snapshot.received_at
        self.quest_last_valid_wrist_at = snapshot.received_at
        self.quest_real_stage = "connecting"
        self.quest_real_confirm_until = 0.0
        self.real_stop_event = threading.Event()
        self._set_button_text(self.quest_real_button, "停止 Quest 实机同步")
        self.quest_status_var.set("Quest 最终确认完成 · 正在连接 30003")
        self.status_var.set("QUEST REAL CONNECTING")
        def work():
            hardware = LiveServoHardware(
                robot_ip,
                max_joint_speed_deg_s=quest_speed,
                feedback=self.monitor,
            )
            try:
                connected_state = hardware.connect()
                if not joint_target_reached(connected_state, home_deg):
                    raise RuntimeError("30003 连接完成时实体 CR3 已不在 Home")
                return hardware, connected_state
            except Exception:
                hardware.close()
                raise

        self._background(
            work,
            self._quest_real_connected,
            self._quest_real_failed,
        )

    def _quest_real_connected(self, result) -> None:
        hardware, state = result
        if self.quest_real_stage != "connecting" or self.estop_latched:
            hardware.close()
            return
        self.live_hardware = hardware
        self.last_applied_live_speed = hardware.max_joint_speed_deg_s
        self.quest_real_stage = "active"
        self._set_sim_from_real(state.joints_deg)
        self.q_target = core.HOME_Q_RAD.copy()
        hardware.start_stream(sim_rad_to_real_deg(self.q_target))
        self.quest_status_var.set(
            "Quest 实机同步 ACTIVE · "
            f"{hardware.max_joint_speed_deg_s:.2f}°/s ServoJ · 腕部丢失自动停止"
        )
        self.status_var.set("QUEST REAL SYNC ACTIVE")
        self.log("Quest 实机同步已启动；当前控制链为 Quest → MuJoCo IK → CR3 ServoJ。")

    def stop_quest_real_sync(self, reason: str | None = None) -> None:
        previous_stage = self.quest_real_stage
        self.real_stop_event.set()
        self.quest_real_stage = "idle"
        self.quest_real_confirm_until = 0.0
        hardware = None
        if previous_stage == "active" and self.live_hardware is not None:
            hardware = self.live_hardware
            self.live_hardware = None
            self.last_applied_live_speed = None
        if hasattr(self, "quest_real_button"):
            self._set_button_text(self.quest_real_button, "Quest 实机同步")
        if hardware is not None:
            self._background(
                hardware.close,
                lambda _=None: None,
                lambda exc: self.log(f"关闭 Quest 30003 失败：{exc}"),
            )
        if previous_stage != "idle":
            self.status_var.set("QUEST SIM READY")
        if reason:
            self.log(reason)

    def _quest_real_failed(self, exc: Exception) -> None:
        if self.quest_real_stage == "idle":
            return
        self.stop_quest_real_sync()
        self.quest_status_var.set(f"Quest 实机同步已中止：{exc}")
        self.status_var.set("QUEST REAL SYNC FAILED")
        self.log(f"Quest 实机同步已中止：{exc}")

    def _check_quest_real_sync(self, now: float) -> None:
        if (
            self.quest_real_stage == "ready"
            and now > self.quest_real_confirm_until
        ):
            self.stop_quest_real_sync("Quest 实机同步第二次确认已超时。")
            return
        if self.quest_real_stage != "active":
            return
        if now - self.quest_last_valid_wrist_at > HAMER_REAL_HAND_WATCHDOG_S:
            self.stop_quest_real_sync(
                f"Quest 腕部数据超过 {HAMER_REAL_HAND_WATCHDOG_S:.1f}s 未更新，实机发送已停止。"
            )

    def on_mode_changed(self) -> None:
        selected = self.mode_var.get()
        if self.teach_active and selected != "teach":
            self.mode_var.set("teach")
            self.log("请先点击“退出示教拖拽”，再切换控制模式。")
            return
        if selected != "live" and self.live_hardware is not None:
            self.stop_live()
        if selected != "hamer" and (self.hamer_tracker is not None or self.hamer_starting):
            self.stop_hamer()
        if selected != "quest" and self.quest_receiver is not None:
            self.stop_quest()
        status_by_mode = {
            "record": "RECORD_PLAYBACK",
            "live": "LIVE_SYNC",
            "teach": "TEACHING READY",
            "hamer": "HAMER SIM READY",
            "quest": "QUEST SIM READY",
        }
        self.status_var.set(status_by_mode.get(selected, "SIMULATION READY"))
        if selected == "hamer":
            self.keyboard_status_var.set(
                self._tr(
                    "HaMeR 仿真：Testing HaMeR=视频测试 · 实时摄像头画面中按 R=设定/重设原点",
                    "HaMeR simulation: use Testing HaMeR for video; press R in live camera to set/reset origin",
                )
            )
        elif selected == "quest":
            self.keyboard_status_var.set(
                self._tr(
                    "Meta Quest 仿真：先启动接收 · 按 R 设定/重设腕部原点 · 仅控制 MuJoCo",
                    "Meta Quest simulation: start receiver · press R to set/reset wrist origin · MuJoCo only",
                )
            )
        else:
            self.keyboard_status_var.set(
                self._tr("键盘待命：W/S=Z  A/D=Y  Q/E=X  I/K=RX  J/L=RY  U/O=RZ")
            )

    def toggle_teach(self) -> None:
        if self.dashboard_action_busy or self.teach_starting:
            self.log("示教模式命令正在执行，请稍候。")
            return
        state = None
        if self.monitor is not None:
            try:
                state = self.monitor.require_fresh(max_age=0.5)
            except Exception:
                state = None
        externally_dragging = bool(
            state is not None and (state.drag_status or state.robot_mode in (6, 8))
        )
        if self.teach_active or externally_dragging:
            self.stop_teach()
        else:
            self.start_teach()

    def start_teach(self) -> None:
        if not self.args.enable_real_execution:
            self.log("示教拖拽已锁定：请使用 --enable-real-execution 启动 GUI。")
            return
        if self.estop_latched or self.real_busy or self.live_hardware is not None:
            self.log("软件急停、真实回放或实时同步活动时不能进入示教拖拽。")
            return
        if self.quest_receiver is not None:
            self.log("Quest 仿真正在运行；请先停止 Quest 接收并切换到录制回放模式。")
            return
        if self.monitor is None:
            self.log("进入示教前请先连接 30004 只读反馈。")
            return
        robot_ip = self._active_robot_ip()
        if robot_ip is None:
            return
        try:
            state = self.monitor.require_fresh(max_age=0.5)
        except Exception as exc:
            self.log(f"无法验证示教前状态：{exc}")
            return
        if state.error or not state.enabled or state.robot_mode != 5:
            self.log(
                "示教拖拽要求机器人无报警、已使能且处于空闲模式 5；"
                f"当前 Mode={state.robot_mode}, Enable={int(state.enabled)}, Error={int(state.error)}。"
            )
            return
        if not messagebox.askyesno(
            "进入示教拖拽",
            "将发送 StartDrag()，机械臂会进入手动拖拽状态。\n\n"
            "请确认负载参数正确、周围无障碍物，并用手可靠扶住机械臂或末端负载。",
            icon="warning",
        ):
            return
        self._clear_motion_keys()
        self.mode_var.set("teach")
        self.dashboard_action_busy = True
        self.teach_starting = True
        self.status_var.set("ENTERING TEACH MODE")
        self.teach_status_var.set("正在请求 StartDrag()...")
        self.log("正在发送 StartDrag()；等待 30004 返回拖拽状态。")
        def work():
            reply = start_drag_mode(robot_ip)
            try:
                state = wait_for_drag_state(self.monitor, True)
            except Exception:
                try:
                    stop_drag_mode(robot_ip)
                except Exception:
                    pass
                raise
            return reply, state

        self._background(
            work,
            self._teach_started,
            self._teach_start_failed,
        )

    def _teach_started(self, result) -> None:
        reply, state = result
        self.dashboard_action_busy = False
        self.teach_starting = False
        self.teach_active = True
        self._set_button_text(self.teach_button, "退出示教拖拽")
        self.teach_status_var.set(
            f"示教拖拽已验证 · Mode={state.robot_mode} · Drag={int(state.drag_status)}"
        )
        self.status_var.set("TEACH MODE ACTIVE")
        self.log(f"StartDrag 已接受：{reply.strip()}")
        self.log("请直接拖动实体机械臂；MuJoCo 将通过 30004 镜像当前姿态。")

    def _teach_start_failed(self, exc: Exception) -> None:
        self.dashboard_action_busy = False
        self.teach_starting = False
        self.teach_active = False
        self.mode_var.set("record")
        self._set_button_text(self.teach_button, "进入示教拖拽")
        self.teach_status_var.set("示教拖拽未启动")
        self.status_var.set("TEACH MODE FAILED")
        self.log(f"StartDrag 失败：{exc}")

    def stop_teach(self) -> None:
        if not self.args.enable_real_execution:
            self.log("退出示教已锁定：请使用 --enable-real-execution 启动 GUI。")
            return
        if self.dashboard_action_busy:
            self.log("另一个控制器命令正在执行，请稍候。")
            return
        if self.recorder.recording:
            self.stop_recording()
        self.dashboard_action_busy = True
        self.status_var.set("EXITING TEACH MODE")
        self.teach_status_var.set("正在请求 StopDrag()...")
        self.log("正在发送 StopDrag()。")
        robot_ip = self.connected_robot_ip
        if robot_ip is None:
            self.log("示教连接 IP 已丢失；请使用 DobotStudio 退出拖拽模式。")
            self.dashboard_action_busy = False
            return
        def work():
            reply = stop_drag_mode(robot_ip)
            state = wait_for_drag_state(self.monitor, False)
            return reply, state

        self._background(
            work,
            self._teach_stopped,
            self._teach_stop_failed,
        )

    def _teach_stopped(self, result) -> None:
        reply, state = result
        self.dashboard_action_busy = False
        self.teach_active = False
        self.teach_starting = False
        self.mode_var.set("record")
        self._set_button_text(self.teach_button, "进入示教拖拽")
        self.teach_status_var.set(
            f"示教拖拽已退出 · Mode={state.robot_mode} · Drag={int(state.drag_status)}"
        )
        self.status_var.set("TEACH MODE STOPPED")
        self.log(f"StopDrag 已接受：{reply.strip()}")
        self.align_sim_to_real()

    def _teach_stop_failed(self, exc: Exception) -> None:
        self.dashboard_action_busy = False
        self.teach_active = True
        self.mode_var.set("teach")
        self._set_button_text(self.teach_button, "重试退出示教")
        self.teach_status_var.set("StopDrag 失败 · 仍按示教活动处理")
        self.status_var.set("STOP DRAG FAILED")
        self.log(f"StopDrag 失败：{exc}；请勿假定机械臂已经退出拖拽状态。")

    def toggle_live(self) -> None:
        if self.live_hardware is not None or self.live_starting:
            self.stop_live()
        else:
            self.start_live()

    def start_live(self) -> None:
        if not self.args.enable_real_execution:
            self.log("实时同步锁定：请使用 --enable-real-execution 启动 GUI。")
            return
        if self.estop_latched or self.real_busy or self.teach_active:
            self.log("软件急停、真实回放或示教拖拽活动时不能启动实时同步。")
            return
        if self.hamer_tracker is not None or self.hamer_starting:
            self.log("当前 HaMeR 仅允许控制 MuJoCo；请先停止 HaMeR 实时仿真再启动实时实机同步。")
            return
        if self.quest_receiver is not None:
            self.log("Quest 仿真正在运行；请先停止 Quest 接收并切换到录制回放模式。")
            return
        if self.monitor is None:
            self.log("正在先建立共享的 30004 反馈；连接完成后请再次点击“启动实时同步”。")
            self.connect_feedback()
            return
        robot_ip = self._active_robot_ip()
        if robot_ip is None:
            return
        try:
            self.monitor.require_fresh(max_age=0.75)
        except Exception as exc:
            self.log(f"30004 反馈尚未恢复，正在重新连接：{exc}")
            self.connect_feedback()
            return
        self.mode_var.set("live")
        try:
            max_speed = self._live_speed_deg_s()
        except ValueError as exc:
            self.log(f"参数无效：{exc}")
            return
        if not messagebox.askyesno(
            "启动实时同步",
            "键盘和画面点动按钮将直接控制实体 CR3。\n\n"
            f"IP: {robot_ip}\n每轴最大速度: {max_speed:.2f} deg/s\n\n"
            "确认工作区安全并启动吗？",
            icon="warning",
        ):
            return
        self.live_starting = True
        self._set_button_text(self.live_button, "取消连接")
        self.status_var.set("STARTING LIVE SYNC")

        def work():
            hardware = LiveServoHardware(
                robot_ip,
                max_joint_speed_deg_s=max_speed,
                feedback=self.monitor,
            )
            try:
                state = hardware.connect()
                return hardware, state
            except Exception:
                hardware.close()
                raise

        self._background(work, self._live_connected, self._live_failed)

    def _live_connected(self, result) -> None:
        hardware, state = result
        if (
            not self.live_starting
            or self.estop_latched
            or self.mode_var.get() != "live"
        ):
            hardware.close()
            return
        self.live_starting = False
        self.live_hardware = hardware
        self.last_applied_live_speed = hardware.max_joint_speed_deg_s
        self._set_sim_from_real(state.joints_deg)
        hardware.start_stream(sim_rad_to_real_deg(self.q_target))
        self._set_button_text(self.live_button, "停止实时同步")
        self.status_var.set("LIVE SYNC ACTIVE")
        self.live_speed_status_var.set(
            f"限速 {hardware.max_joint_speed_deg_s:.2f}°/s · 计划 0.00 · 实际 0.00"
        )
        self.log(
            "实时同步已启动；整个键盘和画面下方按钮现在控制实体 CR3。"
        )

    def _live_failed(self, exc: Exception) -> None:
        self.live_starting = False
        self.live_hardware = None
        self.last_applied_live_speed = None
        self._set_button_text(self.live_button, "启动实时同步")
        self.live_speed_status_var.set("实时同步未启动")
        self.status_var.set("LIVE SYNC STOPPED")
        self.log(f"实时同步停止：{exc}")

    def stop_live(self) -> None:
        if self.hamer_real_stage != "idle":
            self.stop_hamer_real_sync("用户通过实时同步停止了 HaMeR 实机控制。")
            return
        if self.quest_real_stage != "idle":
            self.stop_quest_real_sync("用户通过实时同步停止了 Quest 实机控制。")
            return
        self.live_starting = False
        hardware = self.live_hardware
        self.live_hardware = None
        self.last_applied_live_speed = None
        self._set_button_text(self.live_button, "启动实时同步")
        self.live_speed_status_var.set("实时同步未启动")
        self.status_var.set("LIVE SYNC STOPPED")
        if hardware is not None:
            self._background(hardware.close, lambda _=None: None, self._live_failed)
        self.log("实时同步已停止，本地不再发送 ServoJ。")

    def _tick(self) -> None:
        if self.closed:
            return
        self._drain_ui_events()
        now = time.monotonic()
        dt = float(np.clip(now - self.last_tick, 1e-4, 0.1))
        self.last_tick = now
        self._repeat_motion_keys(now)
        self._update_hamer(now)
        self._update_quest(now)
        self._check_hamer_real_sync(now)
        self._check_quest_real_sync(now)

        if self.armed_until > 0.0:
            remaining = self.armed_until - now
            if remaining <= 0.0:
                self.armed_until = 0.0
                self.arm_status_var.set("武装已过期")
                self.log("真实回放武装已过期。")
            else:
                self.arm_status_var.set(f"真实回放已武装：剩余 {remaining:.1f} 秒")

        try:
            if self.live_hardware is not None:
                hand_real_active = (
                    self.hamer_real_stage == "active"
                    or self.quest_real_stage == "active"
                )
                if self.quest_real_stage == "active":
                    self._apply_quest_real_speed_setting()
                elif not hand_real_active:
                    self._apply_live_speed_setting()
                requested = sim_rad_to_real_deg(self.q_target)
                self.live_hardware.set_stream_target(requested)
                self.live_hardware.raise_stream_error()
                # During robot sync, visualize the smoothed, rate-limited
                # command actually sent to CR3 instead of the raw IK target.
                # This prevents MuJoCo from visibly running ahead of hardware.
                sent_target = real_deg_to_sim_rad(
                    self.live_hardware.latest_command_deg()
                )
                current = self.data.qpos[self.arm_qpos_indices]
                try:
                    sim_speed_scale = self._sim_speed_scale()
                except ValueError:
                    sim_speed_scale = 1.0
                max_step = (
                    core.SIM_MAX_JOINT_SPEED_RAD_S * sim_speed_scale * dt
                )
                self.data.qpos[self.arm_qpos_indices] += np.clip(
                    sent_target - current, -max_step, max_step
                )
            elif self.teach_active and self.monitor is not None:
                actual = self.monitor.require_fresh(max_age=0.5)
                mirrored = real_deg_to_sim_rad(actual.joints_deg)
                self.q_target = mirrored.copy()
                self.data.qpos[self.arm_qpos_indices] = mirrored
            else:
                current = self.data.qpos[self.arm_qpos_indices]
                try:
                    sim_speed_scale = self._sim_speed_scale()
                except ValueError:
                    sim_speed_scale = 1.0
                max_step = (
                    core.SIM_MAX_JOINT_SPEED_RAD_S * sim_speed_scale * dt
                )
                self.data.qpos[self.arm_qpos_indices] += np.clip(
                    self.q_target - current, -max_step, max_step
                )
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
        except Exception as exc:
            if self.live_hardware is not None:
                hamer_real_active = self.hamer_real_stage == "active"
                quest_real_active = self.quest_real_stage == "active"
                hardware = self.live_hardware
                self.live_hardware = None
                self._background(hardware.close, lambda _=None: None, lambda _exc: None)
                if hamer_real_active:
                    self._hamer_real_failed(exc)
                elif quest_real_active:
                    self._quest_real_failed(exc)
                else:
                    self._live_failed(exc)
            elif self.teach_active:
                fault_text = f"示教反馈异常：{exc} · 拖拽状态未知"
                if self.teach_status_var.get() != fault_text:
                    self.log(fault_text)
                self.teach_status_var.set(fault_text)
                self.status_var.set("TEACH FEEDBACK LOST")
            else:
                self.log(f"MuJoCo 更新失败：{exc}")

        self.recorder.sample(
            now,
            self.data.qpos[self.arm_qpos_indices],
            self.data.xpos[self.end_effector_id],
        )
        self._refresh_robot_status()

        if now - self.last_render >= RENDER_PERIOD_S:
            self.last_render = now
            self._render_frame()
        self.root.after(UI_PERIOD_MS, self._tick)

    def _render_frame(self) -> None:
        try:
            self.renderer.update_scene(self.data, camera=self.camera)
            rgb = self.renderer.render()
            image = Image.fromarray(rgb)
            self.photo_image = ImageTk.PhotoImage(image=image)
            if self.canvas_image_id is None:
                self.canvas_image_id = self.canvas.create_image(
                    0,
                    0,
                    image=self.photo_image,
                    anchor=tk.NW,
                )
            else:
                self.canvas.coords(self.canvas_image_id, 0, 0)
                self.canvas.itemconfigure(
                    self.canvas_image_id, image=self.photo_image
                )
            self._render_hamer_preview()
        except Exception as exc:
            self.status_var.set("RENDER ERROR")
            self.log(f"MuJoCo 渲染失败：{exc}")

    def _render_hamer_preview(self) -> None:
        frame = self.hamer_latest_frame
        if self.hamer_tracker is None or frame is None:
            return
        max_width = max(220, min(360, self.canvas.winfo_width() // 3))
        max_height = max(150, min(240, self.canvas.winfo_height() // 3))
        render_key = (self.hamer_latest_frame_sequence, max_width, max_height)
        x = max(8, self.canvas.winfo_width() - 14)
        if (
            render_key == self.hamer_rendered_frame_key
            and self.hamer_preview_image_id is not None
        ):
            self.canvas.coords(self.hamer_preview_image_id, x, 14)
            self.canvas.tag_raise(self.hamer_preview_image_id)
            return
        image = Image.fromarray(frame[:, :, ::-1])
        image.thumbnail((max_width, max_height), Image.Resampling.BILINEAR)
        self.hamer_preview_photo = ImageTk.PhotoImage(image=image)
        self.hamer_rendered_frame_key = render_key
        if self.hamer_preview_image_id is None:
            self.hamer_preview_image_id = self.canvas.create_image(
                x,
                14,
                image=self.hamer_preview_photo,
                anchor=tk.NE,
            )
        else:
            self.canvas.coords(self.hamer_preview_image_id, x, 14)
            self.canvas.itemconfigure(
                self.hamer_preview_image_id,
                image=self.hamer_preview_photo,
            )
            self.canvas.tag_raise(self.hamer_preview_image_id)

    def _refresh_robot_status(self) -> None:
        source = None
        if self.live_hardware is not None:
            source = self.live_hardware.feedback
        elif self.monitor is not None:
            source = self.monitor
        if source is None:
            return
        try:
            state = source.require_fresh(max_age=0.5)
            self.robot_status_var.set(self._format_robot_state(state))
            self._update_feedback_display(state)
            if self.teach_active:
                drag_verified = state.drag_status or state.robot_mode in (6, 8)
                self.teach_status_var.set(
                    "示教拖拽已验证 · Drag=1 · MuJoCo 镜像中"
                    if drag_verified
                    else "StartDrag 已接受 · 尚未看到 Drag=1 / Mode=6"
                )
            if self.live_hardware is not None:
                actual_speed = float(np.max(np.abs(state.joint_speeds_deg_s)))
                self.live_speed_status_var.set(
                    f"限速 {self.live_hardware.max_joint_speed_deg_s:.2f}°/s"
                    f" · 计划 {self.live_hardware.last_planned_speed_deg_s:.2f}"
                    f" · 实际 {actual_speed:.2f}"
                )
        except Exception as exc:
            self.robot_status_var.set(f"反馈异常: {exc}")
            self.io_feedback_var.set(f"控制器遥测：反馈异常 · {exc}")

    def _apply_live_speed_setting(self) -> None:
        """Apply GUI limit edits immediately during an active live session."""
        if self.live_hardware is None:
            return
        try:
            requested_limit = self._live_speed_deg_s()
        except ValueError:
            return
        if (
            self.last_applied_live_speed is not None
            and abs(requested_limit - self.last_applied_live_speed) < 1e-9
        ):
            return
        self.live_hardware.set_max_joint_speed(requested_limit)
        self.last_applied_live_speed = requested_limit
        self.log(f"实时同步限速已更新并立即生效：{requested_limit:.2f} deg/s")

    def _apply_quest_real_speed_setting(self) -> None:
        """Apply Quest speed edits immediately while robot sync is active."""
        if self.live_hardware is None or self.quest_real_stage != "active":
            return
        try:
            requested_limit = self._quest_real_speed_deg_s()
        except ValueError:
            return
        if (
            self.last_applied_live_speed is not None
            and abs(requested_limit - self.last_applied_live_speed) < 1e-9
        ):
            return
        self.live_hardware.set_max_joint_speed(requested_limit)
        self.last_applied_live_speed = requested_limit
        self.log(
            "Quest 实机限速已更新并立即生效："
            f"{requested_limit:.2f} deg/s"
        )

    def _format_robot_state(self, state) -> str:
        mode_name = ROBOT_MODE_NAMES.get(state.robot_mode, "UNKNOWN")
        if state.paused:
            queue_text = "PAUSE"
        else:
            queue_text = "RUN" if state.queue_running else "STOP"
        return (
            f"M{state.robot_mode}/{mode_name} · EN {int(state.enabled)} · "
            f"ERR {int(state.error)} · Q {queue_text}"
        )

    def _update_feedback_display(self, state) -> None:
        joints = np.asarray(state.joints_deg, dtype=float)
        self.joint_feedback_var.set(
            self._tr("关节角度  ", "Joint angles  ")
            + "  ".join(f"J{index + 1} {value:+7.2f}" for index, value in enumerate(joints))
            + " °"
        )
        tcp_pose = state.tcp_pose
        if tcp_pose is None:
            self.tcp_feedback_var.set(
                self._tr(
                    "TCP：当前反馈对象未提供位姿",
                    "TCP: pose is unavailable in the current feedback",
                )
            )
        else:
            tcp = np.asarray(tcp_pose, dtype=float)
            self.tcp_feedback_var.set(
                "TCP  "
                f"X {tcp[0]:+8.2f}  Y {tcp[1]:+8.2f}  Z {tcp[2]:+8.2f} mm  "
                f"RX {tcp[3]:+7.2f}  RY {tcp[4]:+7.2f}  RZ {tcp[5]:+7.2f} °"
            )
        temperatures = state.motor_temperatures_c
        temperature_text = "—"
        if temperatures is not None:
            finite_temperatures = np.asarray(temperatures, dtype=float)
            if np.isfinite(finite_temperatures).any():
                temperature_text = f"{np.nanmax(finite_temperatures):.1f}°C"
        scaling = float(state.speed_scaling)
        scaling_percent = scaling * 100.0 if abs(scaling) <= 1.5 else scaling
        self.io_feedback_var.set(
            self._tr("控制器状态  ", "Controller status  ")
            + self._tr("速度倍率 ", "Speed scale ")
            + f"{scaling_percent:.1f}%  "
            f"Vel/Acc {state.velocity_ratio}/{state.acceleration_ratio}%  "
            + self._tr("最高电机温度 ", "Max motor temp ")
            + f"{temperature_text}  "
            f"DI 0x{state.digital_input_bits:016X}  DO 0x{state.digital_output_bits:016X}"
        )
        torques = state.joint_torques
        currents = state.joint_currents
        if torques is None or currents is None:
            self.torque_feedback_var.set(
                self._tr(
                    "关节力学：当前反馈对象未提供 MActual / IActual",
                    "Joint mechanics: MActual / IActual unavailable",
                )
            )
        else:
            torque_values = np.asarray(torques, dtype=float)
            current_values = np.asarray(currents, dtype=float)
            self.torque_feedback_var.set(
                self._tr("关节力矩 MActual  ", "Joint torque MActual  ")
                + "  ".join(
                    f"J{index + 1} {value:+6.2f}"
                    for index, value in enumerate(torque_values)
                )
                + self._tr("    关节电流 IActual  ", "    Joint current IActual  ")
                + "  ".join(f"J{index + 1} {value:+6.2f}" for index, value in enumerate(current_values))
            )
        tcp_force = state.tcp_force_from_currents
        sensor_force = state.actual_tcp_force
        if tcp_force is None:
            self.force_feedback_var.set(
                self._tr(
                    "TCP / 六维力：当前反馈对象未提供力反馈",
                    "TCP / 6-axis force: force feedback unavailable",
                )
            )
        else:
            names = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
            estimated_text = "  ".join(
                f"{name} {value:+7.2f}"
                for name, value in zip(names, np.asarray(tcp_force, dtype=float))
            )
            if state.six_force_online and sensor_force is not None:
                sensor_text = "  ".join(
                    f"{name} {value:+7.2f}"
                    for name, value in zip(names, np.asarray(sensor_force, dtype=float))
                )
                self.force_feedback_var.set(
                    self._tr("TCP 力（电流估计） ", "TCP force (estimated) ")
                    + estimated_text
                    + self._tr("    六维传感器 ", "    6-axis sensor ")
                    + sensor_text
                )
            else:
                self.force_feedback_var.set(
                    self._tr("TCP 力（电流估计） ", "TCP force (estimated) ")
                    + estimated_text
                    + self._tr(
                        "    六维力传感器：未在线",
                        "    6-axis sensor: offline",
                    )
                )

    def _background(self, function, on_success, on_error) -> None:
        def worker():
            try:
                result = function()
            except Exception as exc:
                self.ui_events.put((on_error, exc))
            else:
                self.ui_events.put((on_success, result))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_ui_events(self) -> None:
        while True:
            try:
                callback, value = self.ui_events.get_nowait()
            except queue.Empty:
                return
            if not self.closed:
                callback(value)

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_entries.append((timestamp, message))
        display_message = (
            translate_runtime_log(message)
            if self.language == "en"
            else message
        )
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {display_message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _render_log_entries(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        for timestamp, message in self.log_entries:
            display_message = (
                translate_runtime_log(message)
                if self.language == "en"
                else message
            )
            self.log_text.insert(tk.END, f"[{timestamp}] {display_message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def clear_log(self) -> None:
        self.log_entries.clear()
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _sim_speed_scale(self) -> float:
        return self._bounded_float(self.sim_speed_var, 0.1, 5.0, "MuJoCo 速度")

    def _translation_step_m(self) -> float:
        return self._bounded_float(
            self.translation_mm_var, 0.1, 50.0, "平移步长"
        ) / 1000.0

    def _rotation_step_rad(self) -> float:
        return math.radians(
            self._bounded_float(self.rotation_deg_var, 0.1, 10.0, "旋转步长")
        )

    def _real_motion_parameters(self) -> tuple[int, int]:
        speed = self._bounded_int(self.real_speed_var, 1, 19, "SpeedJ")
        acceleration = self._bounded_int(self.real_acc_var, 1, 19, "AccJ")
        return speed, acceleration

    def _live_speed_deg_s(self) -> float:
        return self._bounded_float(self.live_speed_var, 0.1, 35.999, "同步限速")

    def _quest_real_speed_deg_s(self) -> float:
        return self._bounded_float(
            self.quest_real_speed_var,
            0.1,
            35.999,
            "Quest 实机限速",
        )

    @staticmethod
    def _bounded_float(variable, low: float, high: float, name: str) -> float:
        try:
            value = float(variable.get())
        except ValueError as exc:
            raise ValueError(f"{name} 必须是数字") from exc
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} 必须在 {low:g} 到 {high:g} 之间")
        return value

    @staticmethod
    def _bounded_int(variable, low: int, high: int, name: str) -> int:
        try:
            value = int(variable.get())
        except ValueError as exc:
            raise ValueError(f"{name} 必须是整数") from exc
        if not low <= value <= high:
            raise ValueError(f"{name} 必须在 {low} 到 {high} 之间")
        return value

    def on_close(self) -> None:
        if self.teach_starting:
            self.log("正在进入示教拖拽，请等待状态验证完成后再关闭 GUI。")
            return
        feedback_dragging = False
        if self.monitor is not None:
            try:
                close_state = self.monitor.require_fresh(max_age=0.5)
                feedback_dragging = bool(
                    close_state.drag_status or close_state.robot_mode in (6, 8)
                )
            except Exception:
                pass
        if self.teach_active or feedback_dragging:
            if self.dashboard_action_busy:
                self.log("示教模式命令正在执行，请稍候再退出。")
                return
            if not messagebox.askyesno(
                "退出示教并关闭",
                "关闭 GUI 前将先发送 StopDrag()，确认退出示教拖拽后再关闭。\n\n继续吗？",
                icon="warning",
            ):
                return
            if self.recorder.recording:
                self.stop_recording()
            self.pending_close = True
            self.dashboard_action_busy = True
            robot_ip = self.connected_robot_ip
            if robot_ip is None:
                self.pending_close = False
                self.dashboard_action_busy = False
                self.log("示教连接 IP 已丢失；GUI 保持打开，请使用 DobotStudio 检查拖拽状态。")
                return

            def work():
                reply = stop_drag_mode(robot_ip)
                state = wait_for_drag_state(self.monitor, False)
                return reply, state

            def teach_stopped_then_close(result) -> None:
                self._teach_stopped(result)
                self._finish_close()

            def teach_close_failed(exc: Exception) -> None:
                self.pending_close = False
                self._teach_stop_failed(exc)
                messagebox.showerror(
                    "无法确认退出示教",
                    f"StopDrag() 失败：\n{exc}\n\nGUI 保持打开，请使用 DobotStudio 检查拖拽状态。",
                )

            self._background(
                work,
                teach_stopped_then_close,
                teach_close_failed,
            )
            return
        if (
            self.real_busy
            or self.live_hardware is not None
            or self.hamer_real_stage in ("homing", "connecting", "active")
            or self.quest_real_stage in ("homing", "connecting", "active")
        ):
            if not messagebox.askyesno(
                "退出并急停",
                "实体运动连接仍处于活动状态。\n退出前将发送软件 EmergencyStop。\n\n继续吗？",
                icon="warning",
            ):
                return
            self.pending_close = True
            if self.estop_latched:
                self._finish_close()
                return
            self.on_emergency_stop()
            return
        self._finish_close()

    def _finish_close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.real_stop_event.set()
        if self.render_resize_job is not None:
            self.root.after_cancel(self.render_resize_job)
            self.render_resize_job = None
        if self.monitor is not None:
            self.monitor.close()
            self.monitor = None
        self.connected_robot_ip = None
        if self.live_hardware is not None:
            self.live_hardware.close()
            self.live_hardware = None
        if self.hamer_tracker is not None:
            self.hamer_tracker.stop()
            self.hamer_tracker = None
        if self.quest_receiver is not None:
            self.quest_receiver.stop()
            self.quest_receiver = None
        self.renderer.close()
        self.root.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Custom GUI for CR3 MuJoCo simulation and guarded hardware control."
    )
    parser.add_argument("--robot-ip", default="192.168.5.11")
    parser.add_argument("--model", type=Path, default=core.DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=core.DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--hamer-bridge-url",
        default="http://128.200.5.196:8765",
        help="HaMeR HTTP bridge base URL; /infer is appended automatically.",
    )
    parser.add_argument(
        "--hamer-video",
        default=str(DEFAULT_HAMER_ORIGIN_VIDEO),
        help="Dedicated HaMeR origin-definition video path.",
    )
    parser.add_argument("--hamer-camera-id", type=int, default=1)
    parser.add_argument(
        "--quest-protocol",
        choices=("udp", "tcp"),
        default="udp",
        help="Meta Quest Hand Tracking Streamer transport.",
    )
    parser.add_argument(
        "--quest-host",
        default="0.0.0.0",
        help="Local address used by the Quest telemetry listener.",
    )
    parser.add_argument("--quest-port", type=int, default=9000)
    parser.add_argument(
        "--quest-hand",
        choices=("right", "left"),
        default="right",
        help="Quest hand whose wrist displacement controls MuJoCo.",
    )
    parser.add_argument(
        "--enable-real-execution",
        action="store_true",
        help="Unlock EnableRobot, playback, and live-sync controls.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = tk.Tk()
    CR3ControlGUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
