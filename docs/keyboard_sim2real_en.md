# CR3 MuJoCo Sim-to-Real Controller

This extension adds a simulation-first CR3 control workflow on top of the
Dobot TCP/IP Python V3 SDK. It provides a custom Tk GUI, MuJoCo rendering,
trajectory recording and guarded playback, low-rate live synchronization,
teaching/drag mode, telemetry, and optional HaMeR hand tracking.

> Real robot motion is opt-in. Without `--enable-real-execution`, normal robot
> state and motion controls remain locked. The red software E-stop remains
> available as a deliberate safety action when a robot IP is configured.

## Requirements

- Python 3.10 or newer
- A sibling `mujoco_ws` workspace containing `scenes/cr3_scene.xml`, or an
  explicit `--model` path
- Dependencies from `requirements-sim2real.txt`
- For real control: a CR3 controller in TCP/IP secondary-development mode

```powershell
python -m pip install -r requirements-sim2real.txt
```

## Start the GUI

Simulation only:

```powershell
python run_keyboard_sim2real_gui.py
```

Unlock guarded real-robot operations:

```powershell
python run_keyboard_sim2real_gui.py `
  --enable-real-execution `
  --robot-ip 192.168.5.11
```

The top-right `ENGLISH / 中文` button switches the visible interface language.

## Controls

| Keys | Cartesian motion |
|---|---|
| `Q / E` | X− / X+ |
| `A / D` | Y− / Y+ |
| `S / W` | Z− / Z+ |
| `K / I` | RX− / RX+ |
| `J / L` | RY− / RY+ |
| `U / O` | RZ− / RZ+ |

The controls also support the numeric keypad and press-and-hold GUI buttons.
Mouse drag rotates/pans the camera and the wheel zooms it.

`Esc` is a global immediate Disable shortcut, including while an input field
has focus. It stops playback, synchronization, teaching, or Robot Home
transmission and sends `DisableRobot()` to the connected robot without a
confirmation dialog. The on-screen Disable button still asks for confirmation.

`Robot Home` checks fresh port 30004 state and displays current/target joints
before confirmation. It then uses the guarded ServoJ path with a fixed 5 deg/s
host limit to return the physical CR3 and MuJoCo to shared Home. Press the same
button again to cancel the move.

## Hardware workflow

1. Put DobotStudio Pro into TCP/IP secondary-development mode.
2. Type or select the robot IPv4 address and click `Connect / Switch`.
3. Power on and enable the robot.
4. Wait until fresh port `30004` feedback reports `Mode=5` and `Enable=1`.
5. If the command queue is stopped, explicitly review it before pressing
   `Continue`—it may resume pending controller commands.
6. Select record/playback, live synchronization, or teaching mode.

The IP field is an editable combo box and remembers successful addresses for
the current run. Editing the text does not silently redirect commands. A new
address becomes active only after `Connect / Switch` closes the old port 30004
receiver and successfully reads the new robot. If the entered and active IPs
differ, new state or motion commands are blocked. Stop, Disable, and software
E-stop still target the active robot so an unfinished edit cannot remove the
ability to stop it. Stop all robot workflows before switching IPs.
Preferably disable the old robot first; switching IP never silently disables it.

The GUI uses one shared `30004` receiver for monitoring, playback, and live
sync. It automatically reconnects after a controller-side reset. Real playback
resamples the recorded timestamps and streams `ServoJ` at about 33 Hz. It does
not stop at intermediate samples and waits for settling only at the final
target. Playback and live sync both use a host-side per-joint speed cap strictly
below 20% of the CR3 rated joint speed.
Playback aborts if fresh feedback falls more than 5 degrees behind the latest
planned command.

The playback SpeedJ percentage is converted into that host-side degree/second
cap. The local SDK documents `AccJ` as applying only to MovJ/JointMovJ, so the
GUI keeps the AccJ field only for compatibility; it is not presented as active
for ServoJ playback.

`Disable Robot` is the normal stop. The red software emergency stop is an
additional software command and never replaces the physical E-stop.

## Teaching and telemetry

Teaching mode sends `StartDrag()` only after checking fresh feedback. MuJoCo
mirrors the physical arm and can record the demonstrated joint trajectory.
Exit teaching with `StopDrag()` before playback or live sync.

Telemetry includes joint position and speed, TCP pose, controller speed ratios,
motor temperature, DI/DO bits, joint torque/current, estimated TCP force, and
six-axis force-sensor data when available.

## HaMeR simulation and real synchronization

HaMeR controls MuJoCo first. The current experiment defaults to
`http://128.200.5.196:8765`; override the bridge URL when needed:

```powershell
python run_keyboard_sim2real_gui.py `
  --hamer-bridge-url http://HAMER_HOST:8765
```

`Testing HaMeR` opens the origin video. Click it again to choose the current
frame as the origin; the remaining video validates the mapping. In live camera
mode, press `R` to set or reset the hand origin. Returning MuJoCo to Home keeps
the hand origin and reanchors only the robot-side origin.

Camera capture and HTTP inference run on separate threads. Inference consumes
only the newest full-resolution, once-mirrored JPEG-80 frame, matching the
original demo input while keeping the camera display responsive.

`HaMeR Robot Sync` has a two-confirmation handover. The first confirmation
aligns MuJoCo from fresh 30004 feedback and moves both CR3 and MuJoCo to the
shared Home at 5 deg/s. A fresh hand result then rebinds the hand origin to the
Home Link6 pose, discarding all simulation-only validation displacement. The
second confirmation verifies that the physical robot remains stopped at Home,
rezeros the current hand pose once more, and only then opens 30003.

Active control streams `HaMeR → MuJoCo IK → rate-limited 33 Hz ServoJ → CR3`.
Stale 30004 feedback, more than 5 degrees of tracking error, five seconds
without a valid hand result, disable, queue pause, or E-stop stops transmission.

## Meta Quest hand-tracking simulation

`Meta Quest` mode directly adapts the UDP listener / threaded TCP server from
the `hand-tracking-streamer` repository's `scripts/sockets.py` and consumes its
documented UTF-8 CSV messages. No HaMeR bridge or APK change is required.

Select the mode, open `Quest Settings`, choose UDP/TCP, port, and controlling
hand, then start the receiver. For UDP, enter this PC's LAN address in the
headset while the GUI normally listens on `0.0.0.0:9000`. Bring the arm to the
Home that matches the selected motion mapping (sim `Home [KP 5]`, or the
guarded robot-sync flow), hold a comfortable neutral pose, and press `R` (or
click `Set Wrist Origin`) to establish the explicit origin.

The Quest panel provides two independent CR3 XYZ mappings, each paired with its
own Home and orientation lock:

- `Original · base XYZ`: keeps the historical mapping
  `(x right, y up, z forward) → (x forward, y left, z up)`, paired with the
  generic Home (J1≈0°, tool faces forward).
- `Reversed tool XYZ`: preserves forward/back motion and flips the transverse
  and vertical axes relative to the original, paired with the screenshot-
  calibrated horizontal-tool Home (J1≈180°, tool rolled about 180° around its
  longitudinal axis).

Both mappings share the same filtering, IK, and host-side speed limiting, but
sim `Home`, `Go Quest Home`, and the guarded Quest robot sync all return to the
Home that matches the current mapping. The tool orientation is locked to that
Home's Link6 pose; wrist XYZ is the only live input, while the wrist quaternion
and 21 landmarks are received for telemetry and do not command end-effector
rotation. Stop the Quest receiver before switching mappings, then press `R`
again to re-establish the wrist origin.

`Go Quest Home` returns MuJoCo to the mode-matching Home at any time. When the
guarded Quest robot sync is already ACTIVE, the button also sends that target
to the physical CR3; otherwise the real robot returns Home only through the
protected `Quest Robot Sync` flow.

To control a CRAFT dexterous hand at the same time, start the Quest receiver
and then click `Start Quest hand follower`. The GUI keeps exclusive use of
UDP `9000`, relays the raw 21-point stream to localhost UDP `9001`, and starts
`CRAFT-Hand_API\python\streamer_thumb_opposition_follow.py`, so CR3 wrist XYZ
and CRAFT finger/thumb opposition run in parallel. The follower previews hand
data only by default; checking `CRAFT hand hardware output (caution)` appends
`--live` for that script. It affects only the CRAFT hand and never changes the
CR3 robot-sync state. The follower buttons do nothing unless the Quest
receiver is running.

The responsive path schedules the newest wrist sample at 60 Hz and tracks wrist
and landmark sequences separately, so a landmark packet cannot reapply an old
wrist position. A velocity-adaptive filter blends from slow alpha 0.30 to fast
alpha 0.85 and normalizes alpha by actual sample time. The old fixed 12 mm IK
step is replaced by a time-based Cartesian tracking speed (0.8 m/s by default,
with at most 50 ms accumulated per update), and the default idle deadzone is
1 mm. These values are editable in `Quest Settings`. Live status reports wrist
input Hz, sample age, wrist speed, and the active filter alpha.

After validating the simulation, `Quest Robot Sync` uses the same guarded
two-confirmation handover as HaMeR. The first confirmation aligns from fresh
30004 feedback and returns CR3 plus MuJoCo to shared Home using the editable
`Quest Speed Limit`. The next
fresh Quest wrist packet establishes a new Home origin, discarding the
simulation-validation offset. A second confirmation rechecks Home, robot
readiness, and wrist freshness before opening 30003 and starting rate-limited
33 Hz ServoJ. The real path still maps XYZ only. Stale feedback, excessive
tracking error, five seconds without wrist data, disable, pause, or E-stop
stops transmission.
The Quest limit defaults to 5 deg/s, accepts 0.1–35.9 deg/s, and updates the
host-side per-joint limiter immediately during ACTIVE synchronization.

## Tests

Tests use mocks and do not send commands to a real robot:

```powershell
python -m unittest discover -s tests -v
```

See [the Chinese guide](keyboard_sim2real.md) for the full safety and operating
notes.
