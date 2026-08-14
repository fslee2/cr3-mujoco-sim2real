# CR3 小键盘仿真、轨迹记录与 Dry Run

入口程序位于 `run_keyboard_sim2real.py`，推荐使用的自定义 GUI 位于
`run_keyboard_sim2real_gui.py`。MuJoCo XML、mesh 默认从相邻的 `mujoco_ws`
读取，也可以通过 `--model` 显式指定 XML。Python 脚本、mapping、轨迹模块和
输出轨迹统一放在 TCP 项目中。

```powershell
conda activate lerobot_mujoco
python -m pip install -r requirements-sim2real.txt
cd TCP-IP-Python-V3
python run_keyboard_sim2real.py
```

默认模式是 `SIM ONLY`。程序不会连接 Dobot，也不会向实体 CR3 发送命令。

仿真响应与实机限速相互独立。当前 MuJoCo 默认每次平移 10 mm、旋转 2 度，
关节在画面中的跟随上限为 60 度/秒；这些数值不会传给实体 CR3。

如需更快的仿真预览，可以只提高 MuJoCo 倍率：

```powershell
python run_keyboard_sim2real.py --sim-speed-scale 2
```

该参数不会改变 Dry Run 或实体命令中的 `SpeedJ=10`、`AccJ=10`。

## 小键盘操作

所有机械臂控制都只接受数字小键盘键码；主键盘数字和字母不会控制机械臂。

| 小键盘按键 | 功能 |
|---|---|
| `8` / `2` | TCP Z+ / Z- |
| `4` / `6` | TCP Y- / Y+ |
| `7` / `9` | TCP X- / X+ |
| `1` / `3` | TCP RX- / RX+ |
| `0` / `.` | TCP RY- / RY+ |
| `-` / `+` | TCP RZ- / RZ+ |
| `5` | 平滑返回仿真 home |
| `Enter` | 开始或停止记录 |
| `/` | 清空内存中的当前轨迹 |
| `*` | Review、安全检查和 JointMovJ Dry Run |

关闭 MuJoCo 窗口即可退出。

## 记录与保存

第一次按小键盘 `Enter` 后以 50 Hz 记录。第二次按下后停止并自动保存到：

```text
TCP-IP-Python-V3/trajectories/cr3_trajectory_YYYYMMDD_HHMMSS.json
```

JSON 记录：

- 相对时间 `time`，单位秒；
- 六轴 `q`，顺序为 `joint1` 至 `joint6`，单位弧度；
- MuJoCo TCP 位置 `tcp`，单位米；
- 当次使用的 joint order、sign、offset 和标定状态。

停止记录后程序进入 REVIEW，不会自动连接或控制实体机器人。

## Dry Run 与安全限制

小键盘 `*` 会检查：

- 轨迹不为空；
- 六轴数据形状正确；
- 不包含 NaN 或 Inf；
- 时间戳严格递增；
- 不超过 MuJoCo joint limits；
- 相邻采样点的最大关节跳变不超过 5 度。

通过后轨迹按至少 0.25 秒间隔降采样，并打印准备使用的：

```text
JointMovJ(J1,J2,J3,J4,J5,J6,SpeedJ=10,AccJ=10)
```

速度和加速度命令默认均为 10%。参数解析和命令生成层都会拒绝任何大于或等于 20% 的值。

## 真实执行锁

共享映射已通过 DobotStudio 笛卡尔运动人工验证，`MAPPING_CALIBRATED` 为
`True`：

```text
cr3_sim2real/joint_mapping.py
```

当前映射为 J1~J6 同序、sign 全 `+1`、offset 全 `0`。第一次实机执行仍只应
录制非常小的单方向轨迹，并使用两次确认流程。

## 两种硬件模式

程序通过启动参数切换模式，避免运行中误按一个键就改变实机控制方式。

### 1. RECORD_PLAYBACK：录制后低速执行

标定完成以后，启动：

```powershell
python run_keyboard_sim2real.py `
  --mode record `
  --enable-real-execution `
  --robot-ip 192.168.5.11
```

启动时仅短暂连接只读反馈端口 `30004`，检查机器人已使能、无错误，
并用实体 CR3 当前关节角初始化 MuJoCo，保证录制起点与实体姿态一致。

操作顺序：

1. 在 DobotStudio 中手动使能 CR3，设置安全的全局速度；
2. 小键盘 `Enter` 开始记录；
3. 用小键盘做一段很小的 MuJoCo 运动；
4. 小键盘 `Enter` 停止并保存；
5. 检查 MuJoCo 姿态和终端 Review；
6. 小键盘 `*` 执行 Safety Check 和 Dry Run；
7. 小键盘 `Enter` 将真实执行武装 15 秒；
8. 15 秒内再按小键盘 `*` 最终确认。

如果超过 15 秒，终端会明确显示 `REAL EXECUTION CONFIRMATION EXPIRED`，
不会打开运动端口或发送命令。此时重新按小键盘 `Enter` 武装即可。

最终确认后程序重新检查 `30004`，随后打开 `30003`。轨迹降采样为关节
waypoints，并依次发送：

```text
JointMovJ(...,SpeedJ=10,AccJ=10)
```

如果反馈中的 `run_queued_cmd=0`，说明控制器执行队列处于停止状态。程序只在
小键盘两次确认全部完成后，通过 `29999` 发送一次 `Continue()`，并等待
`run_queued_cmd=1`；确认队列运行以后才允许发送第一条 `JointMovJ`。程序仍然
不会自动调用 `EnableRobot`、`ClearError` 或自动解除报警。

每条 `JointMovJ` 返回成功后，程序通过 `30004` 的 `QActual` 和实际关节速度
等待当前 waypoint 到位并停止，再发送下一条。默认到位误差为 0.20 度、停止
速度阈值为 0.50 度/秒、单点超时为 60 秒。这样不依赖部分 CR 控制器会拒绝的
`Sync()`，同时避免大量运动命令在控制器中堆积。

如果命令、反馈、机器人状态或到位等待发生异常，程序会停止本次真实回放并
保留 MuJoCo 窗口，不再因为异常直接退出整个程序。

### 2. LIVE_SYNC：MuJoCo 与实体同步移动

标定完成以后，启动：

```powershell
python run_keyboard_sim2real.py `
  --mode live `
  --enable-real-execution `
  --robot-ip 192.168.5.11 `
  --live-max-joint-speed-deg-s 18
```

该模式从 `30004` 的实体姿态初始化 MuJoCo，然后以约 33 Hz 向 `30003`
发送 `ServoJ`。MuJoCo 显示的是 `QActual` 实际反馈，而不是未经执行的目标。

Dobot 手册说明 V3.5.5 以后 `ServoJ` 不受全局速度百分比影响，所以程序在
Python 端对每个关节逐帧限速。CR3 最大关节速度为 180 度/秒，20% 为
36 度/秒；程序要求严格小于 36 度/秒，默认使用 18 度/秒（10%）。

LIVE_SYNC 中只接受小键盘运动键。Home、记录、清空和 Review 键全部禁用。
反馈超过 0.2 秒、机器人未使能、出现错误或状态异常时停止继续发送。

命令行版本不会调用 `EnableRobot`、`ClearError` 或任何自动恢复命令。自定义 GUI
只会在用户明确点击并确认后执行这些控制器操作。

## 自定义 GUI（不使用 MuJoCo 原生 Viewer）

新增入口：

```powershell
python run_keyboard_sim2real_gui.py `
  --enable-real-execution `
  --robot-ip 192.168.5.11
```

GUI 使用 Tk + Pillow 显示 MuJoCo 离屏渲染画面，不打开 MuJoCo 原生 Viewer。
画面支持：

- 左键拖拽旋转相机；
- 右键拖拽平移相机；
- 鼠标滚轮缩放；
- 点击画面取得键盘焦点后，继续使用原有数字小键盘控制；
- GUI 同时支持整个键盘，不再要求必须使用数字小键盘；
- 录制后回放与实时同步两种模式；
- 调节 MuJoCo 速度、平移/旋转步长、SpeedJ、AccJ 和实时同步关节限速。
- 右上角使用 `ENGLISH / 中文` 按钮即时切换界面语言；中文采用微软雅黑 UI，
  英文采用 Segoe UI，遥测会随窗口宽度自动换行。

实机区域提供：

- `连接只读反馈`：只连接 `30004`；
- `仿真对齐实机`：使用当前 `QActual` 初始化 MuJoCo；
- `机器人上电`：确认后向 `29999` 发送 `PowerOn()`；
- `使能 Enable`：确认后向 `29999` 发送 `EnableRobot()`；
- `取消使能`：停止本地回放/同步发送，然后向 `29999` 发送
  `DisableRobot()`，用于日常正常停止；
- `软件紧急停止`：立即锁住本地运动发送并向 `29999` 发送
  `EmergencyStop()`。

软件急停不能替代控制柜或示教器上的实体急停。GUI 不会自动解除急停或恢复；
触发以后必须先人工检查并解除急停。GUI 的“清除报警”只在用户确认后发送一次
`ClearError()`，不会连带执行 `Continue()` 或 `EnableRobot()`。

GUI 还提供以下控制器状态与恢复功能：

- `读取报警`：通过 `GetErrorID()` 读取控制器及 J1—J6 伺服报警，并使用 SDK
  自带的中文报警表显示说明和建议；
- `暂停队列`：先停止本地回放/实时同步发送，再发送 `pause()`；
- `继续队列`：二次确认后发送 `continue()`，因为它可能恢复控制器中尚未完成的命令；
- 30004 实时数据显示 J1—J6、TCP 位姿、控制器倍率、速度/加速度倍率、最高电机
  温度以及 DI/DO 位图。
- GUI、真实回放和实时同步共享同一个 30004 接收器，避免重复连接互相踢线；
  控制器主动断开后接收器会自动重连。反馈恢复前不会发送示教或运动指令。
- `实机全局倍率` 对应控制器 `SpeedFactor()`，必须点击“应用实机全局倍率”并确认；
  GUI 将其限制在 1—19%。它与回放 `SpeedJ/AccJ`、实时主机端 deg/s 限速是三个
  不同层级的参数。

### 示教拖拽与力学反馈

GUI 的第三种控制模式是 `示教拖拽`：

1. 先连接 30004 反馈并确认机器人已使能、无报警、处于空闲 Mode 5；
2. 点击“进入示教拖拽”并确认后，GUI 才发送 `StartDrag()`；
3. 30004 的 `drag_status=1` 或 Mode 6 用于验证控制器确实进入拖拽状态；
4. 示教过程中键盘和 GUI 点动被锁定，MuJoCo 持续镜像实体关节姿态；
5. 此时可点击“开始记录”，把 30004 实体反馈形成示教轨迹；
6. 点击“退出示教拖拽”发送 `StopDrag()`。如果退出失败，GUI 继续按拖拽活动处理，
   不会假定机械臂已经恢复普通模式；示教活动时关闭 GUI 也会先尝试 `StopDrag()`。

示教记录完成后必须先退出拖拽，再进行 Dry Run、武装和真实回放；示教活动期间 GUI
禁止启动任何真实回放命令。

开始拖拽前应确认负载参数正确并可靠扶住机械臂或末端负载。`StartDrag()` 在报警
状态下不可用，且需要控制器支持对应指令。

30004 力学监视新增显示：

- `MActual`：六关节实际力矩反馈；
- `IActual`：六关节实际电流反馈；
- `TCPForce`：由关节电流计算的 TCP 力/力矩；
- `ActualTCPForce`：六维力传感器计算值，仅在 `six_force_online=1` 时显示；
- 原始 `SixForceValue` 和 `JointModes` 也保留在 `RobotFeedback` 中供后续记录。

不同控制器/传感器版本可能对力矩和力反馈采用不同标定。GUI 因此保留协议字段名，
不在未经实机标定的情况下擅自添加 N、N·m 等单位。

日常启动顺序为：`机器人上电` → 等待指示灯稳定 → `使能 Enable`。日常停止
应点击 `取消使能`；只有出现失控、碰撞风险或普通停止无效时才使用红色软件
急停或实体急停。

使能后控制器可能短暂重启 30004 反馈。请等待顶部状态恢复为新鲜的
`M5/ENABLED_IDLE · EN 1`；若队列显示 `Q STOP`，检查没有遗留控制器命令后再
显式点击 `继续队列`。`Continue()` 可能恢复未完成的队列，因此 GUI 不会在进入
示教时静默调用它。

### HaMeR 仿真映射

HaMeR 当前只控制 MuJoCo，不直接向实体机械臂发送命令。默认桥接地址为本机
`http://127.0.0.1:8765`，远程桥接需要显式传入：

```powershell
python run_keyboard_sim2real_gui.py `
  --hamer-bridge-url http://HAMER_HOST:8765
```

`Testing HaMeR` 第一次点击打开原点视频，第二次点击把当前手部位置设为原点，
随后继续播放视频以验证映射；“停止测试视频”可随时结束。实时摄像头模式中按
`R` 设定或重新设定原点。MuJoCo 回到 Home 时保留手部原点，仅把机械臂侧基准
重新锚定到 Home，避免原点丢失。

不带 `--enable-real-execution` 启动时，GUI 仍可完成 MuJoCo 控制、记录、相机
操作和 Dry Run，但 Enable、真实回放和实时同步保持锁定。软件急停按钮始终
保留，以便在 IP 配置正确时发送紧急停止命令。

GUI 键盘映射：

| 按键 | 功能 |
|---|---|
| `W` / `S` | TCP Z+ / Z- |
| `A` / `D` | TCP Y- / Y+ |
| `Q` / `E` | TCP X- / X+ |
| `I` / `K` | TCP RX+ / RX- |
| `J` / `L` | TCP RY- / RY+ |
| `U` / `O` | TCP RZ- / RZ+ |
| 方向键 | Z、Y 平移 |
| `PageUp` / `PageDown` | X+ / X- |
| 主键盘数字 | 与原数字小键盘布局相同 |
| `R` 或 `Enter` | 开始/停止记录 |
| `C` | 清空轨迹 |
| `V` | Dry Run |
| `H` | MuJoCo Home（实时同步模式禁用） |

键盘事件现在绑定在整个 GUI 窗口上，不需要先点击 MuJoCo 画面。只有正在编辑
IP 或参数输入框时会临时屏蔽机械臂运动键；按 `Esc` 即可退出输入状态并恢复
全窗口机械臂控制。

运动键支持按住连续移动、松开立即停止；GUI 失去焦点时会自动清空全部按键
状态，避免按键卡住。MuJoCo 离屏 Renderer 会在窗口缩放后按画布的实际宽高
重新创建，渲染图像与画布保持 1:1 像素和相同宽高比，不拉伸、不裁切。画面
下方也提供 X/Y/Z 和 RX/RY/RZ 的两行按住式控制按钮，可与键盘互换使用。

某些控制器执行 `DisableRobot()` 时会先关闭 `29999` 连接，SDK 因此可能收到
空回复 `b''`。GUI 遇到这种情况会继续检查 `30004`：只有确认
`enable_status=0` 才显示取消使能成功；如果仍为 `1`，则继续报告失败。

`回放 SpeedJ/AccJ` 只作用于录制后的 `JointMovJ` 回放；`实时限速 (deg/s)`
只作用于实时同步的 `ServoJ`。实时同步运行期间修改限速会立即写入主机限速器，
不需要停止再启动。控制模式卡片会显示：

```text
限速 5.00°/s · 计划 5.00 · 实际 4.72
```

限速器使用固定 30 ms 控制周期计算每次最大关节增量，不再把键盘空闲时间或
慢渲染帧累计到下一条命令，因此第一次按键也不会绕过限速。

## DobotStudio 必须切换 TCP/IP 模式

端口能够建立连接不代表控制器已经允许 TCP/IP 运动。在运行带
`--enable-real-execution` 的程序前：

1. 先退出正在运行的 Python 控制程序；
2. 在 DobotStudio Pro 连接实体 CR3；
3. 将机器模式切换为 `TCP/IP 二次开发` / `TCP/IP secondary development`；
4. 等控制器启动完成；
5. 在 DobotStudio 中手动 Enable；
6. 再启动 Python 程序。

如果控制器返回：

```text
-2,{not tcp mode or system is starting}
```

说明仍未处于 TCP/IP 二次开发模式，或切换后控制器还在启动。程序遇到任何
非零 ErrorID 会立即终止剩余 waypoint 并关闭连接。

## 当前必须先做的事

两种实机模式原先由 `MAPPING_CALIBRATED = False` 锁定。现已通过 DobotStudio
笛卡尔运动确认：

- `QActual` 的 J1~J6 顺序；
- 每个 MuJoCo joint 的正负方向；
- 实体零位和 MuJoCo 零位的 offset。

结果已写入：

```text
cr3_sim2real/joint_mapping.py
```

`MAPPING_CALIBRATED` 已改为 `True`，可以进入上述硬件模式；程序仍会执行
使能、报警、反馈 watchdog、起始姿态、轨迹和速度检查。
