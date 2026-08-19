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
| `*` | Review、安全检查和连续 ServoJ Dry Run |

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

通过后轨迹会根据原始时间戳插值为约 33 Hz 的连续关节目标，并打印准备使用的：

```text
t=+0.030s ServoJ(J1,J2,J3,J4,J5,J6,t=0.100,lookahead_time=50,gain=500)
```

回放 SpeedJ 默认为 10%，会在 Python 端换算为 18 度/秒的严格逐帧限速。本地 SDK
明确说明 `AccJ` 只适用于 MovJ/JointMovJ，因此 AccJ 输入目前只为兼容旧界面保留，
不会伪装为对 `ServoJ` 生效。两个字段仍会拒绝大于或等于 20% 的值。

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

最终确认后程序重新检查 `30004`，随后打开 `30003`。轨迹按原始
时间戳插值为约 33 Hz 的关节目标，并连续发送：

```text
ServoJ(...,t=0.100,lookahead_time=50,gain=500)
```

如果反馈中的 `run_queued_cmd=0`，说明控制器执行队列处于停止状态。程序只在
小键盘两次确认全部完成后，通过 `29999` 发送一次 `Continue()`，并等待
`run_queued_cmd=1`；确认队列运行以后才允许发送第一条 `ServoJ`。程序仍然
不会自动调用 `EnableRobot`、`ClearError` 或自动解除报警。

中间采样不再等待机械臂停稳，因此不会在每个轨迹点重新减速。Python 端
依据回放 SpeedJ 对每轴逐帧限速，`30004` 在整个流过程中持续检查使能、报警和
反馈 watchdog；实际关节与已发送目标偏差超过 5 度时中止。只有最后一点会等待位置误差
小于 0.20 度且关节速度小于 0.50 度/秒。

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

- IP 区是可输入的下拉框：可以直接键入新 IPv4，也可以选择本次运行中曾成功连接的
  地址；
- `连接 / 切换`：关闭旧机器人的 `30004`，连接输入框中的新 IP，并且只有连接成功后
  才把全部状态命令、回放、实时同步、示教、HaMeR 和 Quest 控制目标切换过去；
- `断开反馈`：在没有活动实机流程时关闭当前 `30004`；
- `仿真对齐实机`：使用当前 `QActual` 初始化 MuJoCo；
- `机器人上电`：确认后向 `29999` 发送 `PowerOn()`；
- `使能 Enable`：确认后向 `29999` 发送 `EnableRobot()`；
- `取消使能 [Esc]`：按钮按下时先确认；全局 `Esc` 不弹确认框，立即停止
  本地回放、同步、示教或回 Home，并向当前已连接机械臂的 `29999` 发送
  `DisableRobot()`；
- `实机回 Home`：检查 30004 状态并显示当前/目标关节，确认后通过受保护的
  `ServoJ` 流以固定 `5 deg/s` 主机限速返回统一 Home；再次点击可停止；
- `软件紧急停止`：立即锁住本地运动发送并向 `29999` 发送
  `EmergencyStop()`。

只修改 IP 输入框不会改变当前控制对象。输入框和已连接 IP 不一致时，新的运动或状态
命令会被阻止，并提示先点击 `连接 / 切换`；停止、取消使能和软件急停仍指向当前已连接
的机械臂，避免编辑 IP 时失去停止能力。切换前必须先停止回放、实时同步、示教及手部
实机控制。
切换前建议先对旧机械臂执行正常的 `取消使能`；程序不会在切换 IP 时偷偷替用户下使能。

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

### HaMeR 仿真与实机同步

HaMeR 默认先控制 MuJoCo。当前实验默认桥接地址为
`http://128.200.5.196:8765`，也可以显式覆盖：

```powershell
python run_keyboard_sim2real_gui.py `
  --hamer-bridge-url http://HAMER_HOST:8765
```

`Testing HaMeR` 第一次点击打开原点视频，第二次点击把当前手部位置设为原点，
随后继续播放视频以验证映射；“停止测试视频”可随时结束。实时摄像头模式中按
`R` 设定或重新设定原点。MuJoCo 回到 Home 时保留手部原点，仅把机械臂侧基准
重新锚定到 Home，避免原点丢失。

摄像头采集和 HTTP 推理使用两个独立线程。GUI 始终显示最新摄像头帧；HaMeR
推理只取最新的一帧，推理跟不上时直接丢弃中间旧帧，不会积压。送入 HaMeR 的内容
仍与原始演示一致：镜像一次、原分辨率、JPEG 质量 80。

`HaMeR 实机同步` 使用两次明确确认：

1. 第一次确认前检查 30004、使能/报警状态和 TCP/IP 二次开发条件。确认后程序先按
   实机反馈自动对齐 MuJoCo，再以 5 度/秒限速让实体 CR3 与 MuJoCo 回到统一 Home。
2. Home 到位后，验证阶段留下的 MuJoCo 位姿偏移会被丢弃。程序等待一帧新的有效
   `cam_t`，自动把当时手位与 Home Link6 重新绑定。
3. 30 秒内点击 `最终确认 HaMeR 实机同步`。最终确认时再次检查实机仍停在 Home，
   并再次以当前手位归零，然后才打开 30003。
4. 激活后，控制链为 `HaMeR → MuJoCo IK → 5°/s 限速 → 33 Hz ServoJ → CR3`。
   30004 反馈超时、实机跟踪误差超过 5 度、有效手部超过 5 秒未更新、取消使能、
   暂停队列或软件急停都会停止本地发送。

因此仿真验证时 MuJoCo 即使已经离开 Home，也不会把该偏移直接带到实机；实机接管
总是从 `sim = real = Home` 和新的手部原点开始。

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
| `Esc` | 立即停止本地实机发送并执行 `DisableRobot()`，不弹确认框 |

键盘事件现在绑定在整个 GUI 窗口上，不需要先点击 MuJoCo 画面。只有正在编辑
IP 或参数输入框时会临时屏蔽机械臂运动键；`Esc` 在输入框中仍执行全局强制
取消使能，不再用于退出输入状态。

运动键支持按住连续移动、松开立即停止；GUI 失去焦点时会自动清空全部按键
状态，避免按键卡住。MuJoCo 离屏 Renderer 会在窗口缩放后按画布的实际宽高
重新创建，渲染图像与画布保持 1:1 像素和相同宽高比，不拉伸、不裁切。画面
下方也提供 X/Y/Z 和 RX/RY/RZ 的两行按住式控制按钮，可与键盘互换使用。

某些控制器执行 `DisableRobot()` 时会先关闭 `29999` 连接，SDK 因此可能收到
空回复 `b''`。GUI 遇到这种情况会继续检查 `30004`：只有确认
`enable_status=0` 才显示取消使能成功；如果仍为 `1`，则继续报告失败。

`回放 SpeedJ` 用于录制后的连续 `ServoJ` 回放，并转换为 Python 端的严格逐帧
关节限速；`回放 AccJ（保留）` 当前不影响 ServoJ。`实时限速 (deg/s)`
只作用于实时同步的 `ServoJ`。实时同步运行期间修改限速会立即写入主机限速器，
不需要停止再启动。控制模式卡片会显示：

```text
限速 5.00°/s · 计划 5.00 · 实际 4.72
```

限速器使用固定 30 ms 控制周期计算每次最大关节增量，不再把键盘空闲时间或
慢渲染帧累计到下一条命令，因此第一次按键也不会绕过限速。

## Meta Quest 手部跟踪仿真

GUI 的 `Meta Quest 仿真` 直接复用 `hand-tracking-streamer` 原仓库
`scripts/sockets.py` 的 UDP 监听 / TCP 服务端连接方式，以及
`CONNECTIONS.md` 的 UTF-8 CSV 数据格式；不需要 HaMeR 桥接，也不需要修改 Quest APK。

1. 选择 `Meta Quest 仿真`；
2. 在 `Quest 设置…` 中选 UDP/TCP、监听端口和左/右手；
3. UDP 时在 Quest 内填这台电脑的局域网 IP，GUI 默认监听 `0.0.0.0:9000`；
4. 点击 `启动 Quest 接收`，等状态栏出现腕部 XYZ；
5. 点击 `仿真 Home` 或启动实机接管，让 Quest 回到与所选映射模式匹配的 Home；
6. 将手放在舒适的中性位，按 `R` 或点击 `设定腕部原点`；
7. 之后腕部位移通过 `Quest → 坐标转换 → MuJoCo IK` 控制 Link6 的 XYZ，末端姿态锁定为
   该 Home 姿态：`反向末端 XYZ` 为水平工具姿态（TCP Z 轴约平行世界 `+X`），
   `原本·基座 XYZ` 为通用 Home 姿态。

Quest 面板的 `到 Quest Home` 按钮可随时将 MuJoCo 回到与当前映射模式匹配的 Home；
如果 Quest 实机同步已经处于 ACTIVE，则按钮也会发送该目标，否则实体 CR3 需要通过
受保护的 `Quest 实机同步` 流程回 Home。

原始 Unity 坐标按 `(x右, y上, z前) → (x前, y左, z上)` 转换。当前版本只将
腕部 XYZ 映射到 MuJoCo；四元数和 21 个手部关键点已接收并显示状态，
但不用于驱动末端旋转。Quest 实机接管使用与所选映射模式匹配的 Home
（`反向末端 XYZ` 为截图校准的水平工具 Home），不会改变普通键盘/回放模式的通用 Home。

Quest 面板现在提供两种独立的 CR3 XYZ 版本，每个版本各自对应一个 Home：

- `原本·基座 XYZ`：保持历史映射 `(x右,y上,z前) → (x前,y左,z上)`；对应通用
  Home（J1≈0°，末端正前方），方向锁定姿态为该 Home 的 Link6 姿态。
- `反向末端 XYZ`：保留前后方向，将横向和竖直方向相对原映射翻转；对应截图校准
  的水平工具 Home（J1≈180°，末端绕工具轴翻转约 180°）。

两种映射共用滤波、IK 和实机限速，但 `仿真 Home` / `到 Quest Home` / Quest 实机
接管都会回到与当前模式匹配的 Home。建议停止 Quest 接收后切换版本，再重新按
`R` 设定腕部原点。

如果同时需要控制 CRAFT 灵巧手，先启动 Quest 接收，再点击 `启动 Quest 手部跟随`。
GUI 继续独占 Quest 的 `UDP 9000`，将原始 21 点数据转发到本机 `UDP 9001`，并启动
`CRAFT-Hand_API\python\streamer_thumb_opposition_follow.py`：因此 CR3 腕部 XYZ
控制和 CRAFT 手指/拇指对指控制可以并行。默认只做手部数据预览；勾选
`CRAFT 灵巧手实机输出（谨慎）` 后才会给该脚本附加 `--live`，只对 CRAFT 手生效，
不会自动改变 CR3 的实机同步状态。若不启动 Quest 接收，手部跟随按钮不会启动。

Quest 响应链已针对跟手性调整：GUI 以 60 Hz 调度最新腕部帧，wrist 与 landmarks
使用独立序号，因此 landmarks 不会重复驱动旧腕部位置。滤波根据腕部速度在慢速
Alpha（默认 0.30）和快速 Alpha（默认 0.85）之间自适应，并按真实帧间隔归一化；
静止时抑制抖动，快速移动时降低拖尾。固定的 12 mm/帧 IK 步长已改为按时间计算的
笛卡尔追踪速度（默认 0.8 m/s，单次积累最多按 50 ms 计算），静止死区默认 1 mm。
这些参数可在 `Quest 设置…` 中调整。状态栏显示 wrist 输入 Hz、数据年龄、手腕速度和
当前滤波 Alpha，便于区分输入、滤波、IK 与实机限速瓶颈。

在仿真方向、增益和工作空间都验证正确后，`Quest 实机同步` 按钮可进入受保护的
sim-to-real 流程：

1. 第一次确认后用 30004 对齐仿真/实机，并按 GUI 中的 `Quest 实机限速`回统一 Home；
2. 到达 Home 后丢弃仿真验证偏移，用下一帧新鲜 Quest 腕部数据自动重建原点；
3. 30 秒内进行第二次确认，程序再次检查 Home、使能、报警和腕部数据新鲜度；
4. 通过后才打开 30003，以约 33 Hz 发送限速 `ServoJ`。

实机模式仍只使用 Quest XYZ。腕部数据超过 5 秒未更新、30004 反馈过期、跟踪误差
超限、取消使能、暂停队列或软件急停都会停止实机发送。
`Quest 实机限速` 默认为 `5 deg/s`，可在 `0.1–35.9 deg/s` 之间调节；
实机同步 ACTIVE 时修改会立即更新主机端逐关节限速。

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
非零 ErrorID 会立即终止剩余 ServoJ 流并关闭连接。

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
