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

## Hardware workflow

1. Put DobotStudio Pro into TCP/IP secondary-development mode.
2. Power on and enable the robot.
3. Wait until fresh port `30004` feedback reports `Mode=5` and `Enable=1`.
4. If the command queue is stopped, explicitly review it before pressing
   `Continue`—it may resume pending controller commands.
5. Select record/playback, live synchronization, or teaching mode.

The GUI uses one shared `30004` receiver for monitoring, playback, and live
sync. It automatically reconnects after a controller-side reset. Real playback
uses `JointMovJ`; live sync uses `ServoJ` with a host-side per-joint speed cap
strictly below 20% of the CR3 rated joint speed.

`Disable Robot` is the normal stop. The red software emergency stop is an
additional software command and never replaces the physical E-stop.

## Teaching and telemetry

Teaching mode sends `StartDrag()` only after checking fresh feedback. MuJoCo
mirrors the physical arm and can record the demonstrated joint trajectory.
Exit teaching with `StopDrag()` before playback or live sync.

Telemetry includes joint position and speed, TCP pose, controller speed ratios,
motor temperature, DI/DO bits, joint torque/current, estimated TCP force, and
six-axis force-sensor data when available.

## HaMeR simulation

HaMeR controls MuJoCo first; it does not directly command the real robot. Start
a local or remote HaMeR bridge and pass its URL explicitly when needed:

```powershell
python run_keyboard_sim2real_gui.py `
  --hamer-bridge-url http://HAMER_HOST:8765
```

`Testing HaMeR` opens the origin video. Click it again to choose the current
frame as the origin; the remaining video validates the mapping. In live camera
mode, press `R` to set or reset the hand origin. Returning MuJoCo to Home keeps
the hand origin and reanchors only the robot-side origin.

## Tests

Tests use mocks and do not send commands to a real robot:

```powershell
python -m unittest discover -s tests -v
```

See [the Chinese guide](keyboard_sim2real.md) for the full safety and operating
notes.
