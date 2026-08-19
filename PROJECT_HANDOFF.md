# CR3 MuJoCo Sim2Real 项目交接文档

最后整理：2026-08-19  
项目目录：`TCP-IP-Python-V3`  
适用对象：继续开发、现场调试、机械臂联调和问题定位人员

> 本文档是交接基线，不是操作员安全手册。任何实体 CR3 测试都必须有现场人员、实体急停和可控的低速工作区。软件急停不能替代控制柜或示教器上的实体急停。

## 1. 项目目标与边界

本项目在 Dobot TCP/IP Python V3 SDK 上增加了：

- CR3 的 MuJoCo 数字孪生和离屏 GUI；
- 键盘/GUI 笛卡尔点动；
- 轨迹录制、检查、连续 ServoJ 回放；
- MuJoCo 与实体 CR3 的实时同步；
- 30004 实时反馈、报警、关节力矩/电流/TCP 力显示；
- StartDrag/StopDrag 示教拖拽；
- HaMeR 手部位置映射；
- Meta Quest 腕部 XYZ 映射；
- Quest 的原本基座映射与反向末端映射；
- Quest 数据并行转发给 CRAFT 灵巧手跟随脚本。

当前项目的核心原则是：

1. 默认只运行 MuJoCo，不发送实体命令。
2. 实体命令必须显式使用 `--enable-real-execution`，并通过 GUI 的确认流程。
3. `30004` 用于读取真实状态和安全监控；除 Quest/实时同步明确启用外，MuJoCo 不反向驱动实体。
4. 任何控制手感问题都必须区分输入目标、滤波目标、限速目标和实体反馈，不能只看一行 `Send ServoJ` 日志下结论。

## 2. 代码结构

```text
TCP-IP-Python-V3/
├─ run_keyboard_sim2real.py       # 命令行仿真、录制、Dry Run、回放入口
├─ run_keyboard_sim2real_gui.py   # 当前主要 GUI 入口
├─ dobot_api.py                   # 原始 Dobot TCP/IP SDK 封装
├─ cr3_sim2real/
│  ├─ hardware.py                 # 30004、29999、30003 和 ServoJ 控制层
│  ├─ joint_mapping.py             # 实体角度 ↔ MuJoCo 弧度映射
│  ├─ trajectory.py                # 记录、校验、重采样、Dry Run
│  ├─ quest_hand.py                # Quest 协议、接收、坐标映射和滤波
│  ├─ hamer_bridge.py              # HaMeR HTTP 桥接和手部映射
│  └─ hamer_real.py                # HaMeR 实机安全计划/预览
├─ tests/
│  └─ test_keyboard_sim2real.py    # 单元和控制流程测试
├─ docs/
│  └─ keyboard_sim2real.md         # 面向使用者的详细说明
├─ DEBUGGING_EXPERIENCE.md         # 控制顿挫问题的回归与诊断记录
└─ trajectories/                   # 录制的 JSON 轨迹
```

模型默认从相邻的 `mujoco_ws` 目录读取；需要换模型时使用 `--model`，不要在代码中硬编码新的 XML 路径。

## 3. 控制链路

### 3.1 普通键盘/GUI实时控制

```text
键盘或GUI点动
  → q_target（MuJoCo关节目标，rad）
  → sim_rad_to_real_deg（仅在实机模式）
  → LiveServoHardware.set_stream_target
  → 固定周期 ServoJ worker
  → 30003 / DobotApiMove.ServoJ
  → CR3
```

没有 `live_hardware` 时，GUI 只让 MuJoCo 的 `qpos` 逐步靠近 `q_target`；不会打开 30003，也不会发送实体运动命令。

### 3.2 轨迹回放

```text
录制 MuJoCo q
  → TrajectoryRecorder
  → validate_trajectory
  → resample_joint_trajectory
  → ServoJ 连续目标
  → 两次确认、反馈检查、队列检查
  → 30003
```

回放不应使用每个点一次 `JointMovJ` 再等待到位的方式。那会产生“一个点一个点走、每点停顿”的观感。连续轨迹应使用原始时间戳重采样后连续发送 ServoJ。

### 3.3 实时同步

实时同步是“输入设备 → MuJoCo IK → q_target → 实体 ServoJ”的目标流；它不是把实体反馈作为 IK 闭环误差控制器。30004 反馈主要用于：

- 初始化或对齐 MuJoCo；
- 检查使能、报警、模式和反馈新鲜度；
- 监视实际关节与已发送目标的偏差；
- 在断线、超时或危险状态时停止发送。

GUI 会根据实际已发送命令更新可视化，避免画面明显跑在实体前面。

### 3.4 示教拖拽

```text
29999 StartDrag()
  → 30004 验证 drag_status / robot_mode
  → 人工拖动实体 CR3
  → GUI 镜像 QActual 到 MuJoCo
  → 29999 StopDrag()
```

示教期间 GUI 不发送 ServoJ。退出示教必须等到 30004 确认已离开拖拽状态，不能只根据 `StopDrag()` 返回字符串判断。

## 4. Quest与CRAFT并行链路

### 4.1 Quest输入

Quest Hand Tracking Streamer 通过 UDP 默认发到电脑 `9000` 端口。`quest_hand.py` 支持：

- `Right/Left wrist`：位置 3 个数 + 四元数 4 个数；
- `Right/Left landmarks`：21 个三维关键点，共 63 个数；
- 可选帧号和设备时间戳。

当前 CR3 控制只使用腕部位置 XYZ。四元数暂不驱动末端旋转；landmarks 用于状态显示和 CRAFT 并行手部控制。

### 4.2 两种Quest运动版本

GUI 中的 `Quest 运动映射` 有两个版本：

- `原本·基座 XYZ`：

  ```text
  Unity (x右, y上, z前) → CR3 (x前, y左, z上)
  ```

- `反向末端 XYZ`：保留前后方向，将横向和竖直方向相对原映射翻转，适配水平工具 Quest Home 中工具绕纵向轴翻转约 180° 的情况。

实现常量位于 `cr3_sim2real/quest_hand.py`：

```python
R_UNITY_TO_ROBOT
R_UNITY_TO_ROBOT_REVERSED_END
QUEST_MOTION_MODE_ORIGINAL
QUEST_MOTION_MODE_REVERSED_END
```

两种版本共用滤波、IK 和实体限速，但每个版本各自对应一个 Home（决策函数
`quest_home_q_rad_for_mode` 位于 `run_keyboard_sim2real.py`）：

- `原本·基座 XYZ` → 通用 Home `HOME_Q_RAD`（J1≈0°，末端正前方）；
- `反向末端 XYZ` → 水平工具 Home `QUEST_HOME_Q_RAD`（J1≈180°）。

`仿真 Home`、`到 Quest Home`、Quest 实机接管和方向锁定都会随当前映射模式选择对应
Home。切换映射时若已有原点，程序会清除原点，必须重新按 `R`，防止符号翻转造成
目标跳变；若机械臂停在旧 Home，需重新按 `到 Quest Home` 切换。

### 4.3 Quest实机接管

Quest 实机接管不是“直接把 MuJoCo 当前偏移发送给 CR3”，而是：

1. 读取 30004，验证机器人状态；
2. MuJoCo 与实体 CR3 回到统一 Quest Home；
3. 等待新的 Quest 腕部帧；
4. 使用该帧重新建立真实控制原点；
5. 第二次确认后才连接 30003；
6. 通过主机端关节速度限制连续发送 ServoJ。

这样可以丢弃仿真验证阶段造成的偏移，避免 MuJoCo 和实体 Home 不一致时突然跳动。

### 4.4 CRAFT灵巧手并行控制

CRAFT 脚本位于另一个仓库：

```text
E:\material_for_uci_experiment\CRAFT-Hand_API
```

脚本：

```text
python\streamer_thumb_opposition_follow.py
```

GUI 负责占用 Quest 的 `UDP 9000`，并将原始文本转发到本机 `UDP 9001`，然后启动 CRAFT streamer。因此可以同时实现：

```text
Quest wrist       → CR3 末端 XYZ
Quest landmarks   → CRAFT 手指/拇指对指
```

GUI 中 CRAFT 实机输出默认关闭。只有勾选 `CRAFT 灵巧手实机输出（谨慎）` 后，子进程才会加 `--live`。该开关只影响 CRAFT 手，不会自动打开 CR3 实机同步。

不要在 GUI 已启动 Quest 接收时，再手动让 CRAFT 脚本监听 `9000`；两个进程会抢同一个 UDP 端口。手动测试时应停止 GUI Quest 接收，或者使用 GUI 的本地转发流程。

## 5. 网络、端口和硬件前置条件

### 5.1 CR3端口

| 端口 | 用途 | 代码使用方式 |
|---:|---|---|
| 29999 | Dashboard：Enable、Disable、PowerOn、报警、队列、拖拽 | `hardware.py` 按命令短连接 |
| 30003 | 运动命令：ServoJ / MovJ 等 | 实时运动期间保持连接 |
| 30004 | 1440 字节实时反馈 | `FeedbackReceiver` 共享长连接 |

当前现场常用实体 IP 为 `192.168.5.11` 和 `192.168.5.12`。电脑网卡必须与目标控制器在同一网段，先用 `ping` 和端口连通性确认，不要只看 GUI 输入框中的字符串。

### 5.2 控制器状态

实体测试前必须：

1. DobotStudio Pro 切换到 `TCP/IP 二次开发`；
2. 等待控制器完成启动；
3. 在 DobotStudio 中确认 Enable、无报警、工作区安全；
4. 再启动带 `--enable-real-execution` 的程序。

若收到：

```text
-2,{not tcp mode or system is starting}
```

优先检查 TCP/IP 二次开发模式和控制器启动状态，不要通过重复发命令解决。

### 5.3 软件环境

最低建议环境：

- Windows 10/11 64 位；
- Python 3.12.x；
- Conda 环境名：`lerobot_mujoco`；
- `numpy >= 1.26`；
- `mujoco >= 3.2`；
- `Pillow >= 10.0`；
- `opencv-python >= 4.9`；
- 可访问 MuJoCo XML 和 mesh 的工作区；
- CRAFT 仓库自己的 `.venv`，包含其硬件依赖。

安装仿真依赖：

```powershell
conda activate lerobot_mujoco
python -m pip install -r requirements-sim2real.txt
```

CRAFT 依赖不要混装到 CR3 Conda 环境；优先使用：

```powershell
E:\material_for_uci_experiment\CRAFT-Hand_API\.venv\Scripts\python.exe
```

## 6. 已遇到的问题、原因与解决方案

### 6.1 实机“不动”但命令返回成功

现象可能是：`JointMovJ` 返回 `0,{}`，但机械臂没有明显动作。

检查顺序：

1. 是否在 TCP/IP 二次开发模式；
2. 是否已 Enable、无报警；
3. 30004 的 `enable_status`、`robot_mode`、`run_queued_cmd`；
4. 是否发送了过小的关节差；
5. 是否只有 MuJoCo 目标改变而没有打开 30003。

命令返回成功只表示控制器接受了文本，不等于实体已经完成运动。

### 6.2 `Sync()` 返回 `ErrorID -1`

`Sync()` 属于队列命令语义，不能作为 ServoJ 每一帧的等待机制。若队列停止，应在明确确认安全、没有残留命令后通过 GUI 的“继续队列”恢复；不要在连续 ServoJ 流中插入逐点 `Sync()`。

### 6.3 `DisableRobot` 返回空字节 `b''`

部分控制器会在处理 DisableRobot 时主动关闭 29999 连接，SDK 于是收到空回复。解决方案是：

- 不把空回复直接当作“肯定成功”；
- 关闭旧 Dashboard socket；
- 用 30004 检查 `enable_status` 是否变为 0；
- 如果仍为 1，记录失败并让现场人员在 DobotStudio 检查。

日常停止优先使用 GUI 的取消使能；危险情况使用实体急停或软件 EmergencyStop。

### 6.4 30004 超时、10054或“远程主机强迫关闭”

常见原因：控制器重启、Enable/Disable 触发短暂反馈重连、多个程序竞争 30004、DobotStudio 与 Python 同时抢控制资源。

当前方案是 GUI/回放/实时同步共享 `FeedbackReceiver`，不重复创建反馈连接。出现异常时先停止其他客户端，等待反馈恢复，再执行重新连接；不要并行启动多个 GUI。

### 6.5 实机顿挫、震动或来回反向

已知平滑基线：

```text
2fc5b230b39343a7fa1b597036a588a71ee9a5d2
```

历史回归表明，以下参数组合可能明显影响手感：ServoJ `t`、`lookahead_time`、目标 EMA、限速时间步长和命令发送周期。不要同时修改多个参数，也不要把 Quest、IK 和底层 ServoJ 混为一个问题。

诊断时必须同时记录：

```text
timestamp
send_dt
servo_rtt
requested_target
filtered_target
planned_target
actual_feedback
tracking_error = actual_feedback - planned_target
```

判断原则：

- `requested_target` 已经来回跳：问题在输入、Quest 映射或 IK；
- requested 平滑、planned 出现固定步长：主机限速或滤波在起作用；
- planned 平滑、actual 跟不上：实机速度/队列/通信/负载问题；
- actual 误差很小但仍震动：重点检查 ServoJ 参数、控制器轨迹执行和机械负载。

回归实验每次只改一个变量，并保留 Git 提交和完整日志。

### 6.6 速度参数“不起作用”

必须区分：

- MuJoCo 倍率：只影响仿真显示/跟随；
- `SpeedJ` / `AccJ`：主要对应 MovJ/JointMovJ，不能假定对 ServoJ 同样生效；
- 实时同步的主机 `deg/s` 限速：当前连续 ServoJ 的主要安全速度约束；
- 控制器 `SpeedFactor()`：控制器全局倍率，必须单独确认。

不要用提高 MuJoCo 倍率来推断实体速度，也不要为了追求跟手而删除主机限速。

### 6.7 Quest方向、末端姿态或“下不去”

Quest 原点、Unity→CR3 坐标变换、Link6 body 原点、Studio 自定义 TCP site 不是同一件事。先确认：

1. 末端使用的是 MuJoCo `Link6` body 还是自定义 site；
2. Studio 当前工具坐标系和 TCP 偏移；
3. Quest Home 的关节角、末端欧拉角和关节限位；
4. 选择的是原本映射还是反向末端映射；
5. 是否在切换映射后重新按了 `R`。

滤波不能修复 TCP/site 的固定几何偏移。不要为了“能到达”随意扩大 MuJoCo joint limits，也不要把真实不可达姿态直接硬塞给 ServoJ。

### 6.8 MuJoCo穿模或运动手感突然改变

当前控制主要是设置关节目标并调用 `mj_forward`，不是对整个场景做动力学仿真。不要未经基线实验把控制循环改成 `mj_step`，也不要用动力学积分替代现有目标跟随；这会改变接触、惯性和碰撞表现，且可能让仿真与现有 IK/实机链路不一致。

穿模首先检查 XML 中的 collision geom、工作台位置和当前 qpos；不要通过改变实时控制层来“修”模型几何问题。

## 7. 安全操作流程

### 7.1 只做仿真

```powershell
conda activate lerobot_mujoco
cd E:\material_for_uci_experiment\05_CR3控制项目\TCP-IP-Python-V3
python run_keyboard_sim2real_gui.py
```

确认日志显示没有发送实体命令。先完成 MuJoCo 方向、Home、工作空间和键盘测试。

### 7.2 启用实体功能

```powershell
python run_keyboard_sim2real_gui.py `
  --enable-real-execution `
  --robot-ip 192.168.5.11
```

推荐顺序：

1. 输入或选择 IP；
2. 连接/切换；
3. 读取 30004；
4. 在 DobotStudio 确认上电、Enable、无报警；
5. 先做低速、短行程验证；
6. 先仿真，再进入受保护的实机同步确认；
7. 随时准备使用 Esc 取消使能或实体急停。

切换 IP 前必须停止回放、实时同步、示教、HaMeR 实机和 Quest/CRAFT 实机流程。只编辑 IP 输入框不会自动切换当前控制对象。

### 7.3 Quest测试

1. Quest streamer 目标设置为电脑局域网 IP、UDP、端口 9000；
2. GUI 启动 Quest 接收；
3. 选择映射版本；
4. 先将 MuJoCo 到 Quest Home；
5. 手放在中性位，按 `R`；
6. 只验证 MuJoCo；
7. CRAFT 先使用预览模式；
8. 实机同步必须经过两次确认，且实时同步限速先从低值开始。

## 8. 测试和验收

每次改动后至少执行：

```powershell
conda run -n lerobot_mujoco python -m unittest discover -s tests -q
conda run -n lerobot_mujoco python -m py_compile `
  cr3_sim2real\quest_hand.py `
  run_keyboard_sim2real_gui.py
git diff --check
```

当前交接版本的测试基线为 58 项通过。涉及 Quest 映射时必须覆盖：

- 原本映射的 XYZ 符号；
- 反向末端映射的 XYZ 符号；
- 原点清除和重新标定；
- 数据过期和无有效 wrist 的安全行为。

涉及 ServoJ 时还要在 fake controller 或空载低速环境验证，不得只依靠 GUI 画面判断。

实体验收最小记录应包含：Git commit、机器人 IP、控制器状态、模式、Quest 映射、速度限制、发送周期、实际反馈和异常日志。

## 9. 编码规范

### 9.1 分层原则

- `hardware.py`：只负责协议、连接、反馈和硬件安全，不写 Tk 控件逻辑。
- `quest_hand.py`：只负责 Quest 协议、数据快照、坐标变换和滤波，不直接调用 Dobot API。
- `trajectory.py`：只负责轨迹数据和校验，不打开 socket。
- `joint_mapping.py`：只负责角度单位和关节映射，不承担限速。
- `run_keyboard_sim2real_gui.py`：负责界面、状态机和把模块组合起来。
- `run_keyboard_sim2real.py`：保持命令行和核心数学函数可独立测试。

新增功能应优先放入合适的模块，再由 GUI 调用；不要在按钮回调中复制一套 socket、滤波或 IK。

### 9.2 单位与坐标

变量名必须体现单位或类型：

```python
q_rad
joints_deg
speed_deg_s
position_m
position_mm
timestamp_s
```

明确写出坐标系名称，例如 `unity_position`, `robot_delta_m`, `link6_position_m`。不要使用未说明的 `x/y/z`、`speed` 或 `origin`。

### 9.3 错误与日志

- 网络、协议、参数错误使用明确异常类型或带命令名的 `RuntimeError`。
- 所有真实命令都记录目标 IP、端口、命令名和回复摘要。
- 日志内容必须同时加入中文和英文翻译映射，保证 GUI 切换 English 后仍可读。
- 不吞掉控制错误；只有 CRAFT 本地转发这类非核心旁路可以 best-effort 丢弃。
- 后台线程不得直接操作 Tk；使用 `ui_events` 回主线程。

### 9.4 安全状态机

新增运动入口必须检查：

- `estop_latched`；
- 是否已有其他真实运动流程；
- `--enable-real-execution`；
- 30004 是否新鲜；
- 机器人是否使能、无报警、模式正确；
- 输入目标是否在 joint limits 和速度限制内。

停止路径必须可重复执行，且不能依赖窗口焦点、子进程仍存活或某个单次回复。

## 10. 命名准则

| 对象 | 规则 | 示例 |
|---|---|---|
| Python 文件 | 小写下划线 | `quest_hand.py`, `hamer_bridge.py` |
| 类 | PascalCase | `QuestWristMapper`, `FeedbackReceiver` |
| 函数/方法 | snake_case，动词开头 | `start_quest`, `require_motion_ready` |
| 常量 | 全大写下划线 | `QUEST_CONTROL_RATE_HZ` |
| Tk 变量 | 以 `_var` 结尾 | `quest_speed_var` |
| 角度 | `_rad` 或 `_deg` | `home_q_rad`, `joints_deg` |
| 速度 | `_deg_s` 或 `_m_s` | `max_joint_speed_deg_s` |
| 布尔量 | `is_`, `has_`, `enable_`, `*_active` | `has_wrist`, `teach_active` |
| 测试 | `test_<行为>_<预期>` | `test_quest_wrist_mapper_uses_unity_to_robot_axes` |
| Git提交 | 一个逻辑改动一个提交 | `Fix Quest origin re-anchoring` |

禁止使用 `tmp2`、`new_mapper`、`speed2`、`data1` 这类无法表达语义的名字。涉及坐标变换时，命名必须说明输入和输出坐标系。

## 11. Git与调试纪律

1. 调试前先记录 `git status` 和当前 commit。
2. 先运行已知平滑基线，再改一个变量。
3. 不提交 `__pycache__`、临时轨迹、现场 IP 密钥或硬件日志中的敏感信息。
4. 不使用 `git reset --hard` 覆盖用户改动；回档前先创建可恢复分支或提交。
5. 每个控制参数实验都要写清：基线、唯一改动、测试动作、日志结论和回滚点。
6. 代码、文档和测试一起更新；不要只改 GUI 文案而不更新安全说明。

历史重要节点：

```text
2fc5b230  已知实时同步平滑基线
9f2a986  恢复到 2fc5b230 控制基线
801c562  记录实时同步回归诊断
9fc9475  Quest 腕部姿态可选支持
416de73  Quest 水平固定末端姿态
e971d0a  Quest Home 和 Quest Home 按钮
```

当前工作区如果还有未提交的 Quest 双映射/CRAFT 转发改动，交接前应先单独提交并在本文档顶部补充 commit，避免后续误认为它已经进入主分支。

## 12. 交接前检查清单

- [ ] `PROJECT_HANDOFF.md`、`DEBUGGING_EXPERIENCE.md` 和 `docs/keyboard_sim2real.md` 内容一致。
- [ ] 测试全部通过，`git diff --check` 无错误。
- [ ] MuJoCo XML、mesh 和默认路径在新机器上可访问。
- [ ] Conda 环境和 CRAFT `.venv` 分离且可启动。
- [ ] 两个 CR3 IP 已通过 ping/端口测试。
- [ ] DobotStudio TCP/IP 二次开发模式已确认。
- [ ] 30004 只读反馈能稳定读取至少数十秒。
- [ ] 仿真 Home、普通键盘、回放 Dry Run 和 Quest 原点均通过。
- [ ] 原本 Quest 映射和反向末端 Quest 映射都重新验证过。
- [ ] CRAFT 先预览、后低幅度实机，且没有与 GUI 抢 UDP 9000。
- [ ] 实体测试现场确认实体急停可用，软件急停不作为唯一安全措施。

## 13. 后续建议

优先级建议如下：

1. 将当前未提交的双 Quest 映射和 CRAFT 转发代码形成独立 Git 提交。
2. 为 `LiveServoHardware` 补充 requested/filtered/planned/actual 四路诊断日志。
3. 用固定采样数据分别验收原本 Quest 映射和反向末端映射，不依靠主观手感。
4. 在没有实体手部硬件时只保留 CRAFT 预览，不把 `--live` 作为默认值。
5. 若需要控制 CR3 末端夹爪，应另行定义 CR3 工具/IO 协议；不要把 CRAFT 手部脚本误认为 CR3 夹爪控制器。

