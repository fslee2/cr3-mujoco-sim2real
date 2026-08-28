"""Standalone CRAFT Hand position GUI.

This small utility intentionally does not import or start MediaPipe.  It is
for the safe, repeatable workflow used when teaching hand poses:

    connect/preflight (torque remains off) -> manually pose -> save ->
    enable torque -> call a saved pose

The saved files use the same ``trajectories/craft_hand_positions`` directory
and JSON format as the main CR3 GUI.  A saved pose is sent as one complete
position target; the Dynamixel position controller performs the continuous
move, rather than the GUI issuing step-by-step jog commands.

Example::

    python craft_hand_pose_gui.py --port COM11 --baud 57600

The default port is COM11 because that is the CRAFT Hand bus.  The wrist
motors in ``wrist_jog.py`` use a separate bus (for example COM13) and are not
handled by this program.
"""

from __future__ import annotations

import argparse
import importlib
import json
import queue
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np


MOTOR_IDS = list(range(1, 16))
POSITION_FORMAT = "craft_hand_motor_position_v1"
DEFAULT_PORT = "COM11"
DEFAULT_BAUD = 57600
DEFAULT_MOVE_DURATION_S = 2.0


def default_api_root() -> Path:
    """Return the sibling CRAFT-Hand_API checkout used by this project."""
    return Path(__file__).resolve().parents[2] / "CRAFT-Hand_API"


def load_craft_api(api_root: Path):
    """Import the CRAFT safety harness from the external repository."""
    python_dir = api_root / "python"
    if not python_dir.is_dir():
        raise FileNotFoundError(f"CRAFT-Hand API python directory not found: {python_dir}")
    path_text = str(python_dir)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    module = importlib.import_module("craft_hand_utils.safety_harness")
    return module.SafeDynamixelHarness


def read_pose(path: Path) -> tuple[list[int], np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != POSITION_FORMAT:
        raise ValueError(f"Unsupported hand-pose format in {path.name}")
    motor_ids = [int(value) for value in data.get("motor_ids", [])]
    positions = np.asarray(data.get("positions_rad", []), dtype=np.float64).reshape(-1)
    if motor_ids != MOTOR_IDS or positions.size != len(MOTOR_IDS):
        raise ValueError("The saved pose must contain motors M1-M15")
    if not np.all(np.isfinite(positions)):
        raise ValueError("The saved pose contains a non-finite position")
    return motor_ids, positions


def write_pose(path: Path, positions: np.ndarray) -> None:
    values = np.asarray(positions, dtype=np.float64).reshape(-1)
    if values.size != len(MOTOR_IDS) or not np.all(np.isfinite(values)):
        raise ValueError("Cannot save an incomplete motor pose")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": POSITION_FORMAT,
        "name": path.stem,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "motor_ids": MOTOR_IDS,
        "units": {"position": "rad"},
        "positions_rad": values.tolist(),
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class HandPoseGUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.root.title("CRAFT Hand · Position Manager")
        self.root.geometry("820x620")
        self.root.minsize(680, 500)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.SafeDynamixelHarness = None
        self.harness = None
        self.torque_enabled = False
        self.operation_lock = threading.Lock()
        self.operation_thread: threading.Thread | None = None
        self.operation_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self.port_var = tk.StringVar(value=args.port)
        self.baud_var = tk.StringVar(value=str(args.baud))
        self.status_var = tk.StringVar(value="未连接；点击“连接 / 自检”开始。")
        self.pose_var = tk.StringVar(value="选择已保存位置…")
        self.position_var = tk.StringVar(value="当前位置：--")
        self.torque_var = tk.StringVar(value="扭矩：未使能")
        self.pose_files: dict[str, Path] = {}
        self.connection_button: ttk.Button | None = None
        self.enable_button: ttk.Button | None = None
        self.disable_button: ttk.Button | None = None
        self.save_button: ttk.Button | None = None
        self.use_button: ttk.Button | None = None

        self._build_ui()
        self.refresh_pose_list()
        self.root.after(100, self._poll_events)
        self.root.after(250, self._poll_position)

    @property
    def positions_dir(self) -> Path:
        return Path(self.args.positions_dir).resolve()

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Hint.TLabel", foreground="#536273")
        style.configure("Danger.TButton", foreground="#9c1c1c")

        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(5, weight=1)
        ttk.Label(outer, text="CRAFT Hand 位置管理", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            outer,
            text="不启动 MediaPipe；只用于掉电手动摆位、保存和连续调用位置。",
            style="Hint.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 14))

        connection = ttk.LabelFrame(outer, text="连接", padding=10)
        connection.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        connection.columnconfigure(1, weight=1)
        ttk.Label(connection, text="串口").grid(row=0, column=0, sticky="w")
        ttk.Entry(connection, textvariable=self.port_var, width=14).grid(
            row=0, column=1, sticky="ew", padx=(8, 12)
        )
        ttk.Label(connection, text="波特率").grid(row=0, column=2, sticky="w")
        ttk.Entry(connection, textvariable=self.baud_var, width=10).grid(
            row=0, column=3, sticky="w", padx=(8, 12)
        )
        self.connection_button = ttk.Button(
            connection, text="连接 / 自检", command=self.connect
        )
        self.connection_button.grid(row=0, column=4, sticky="ew")

        safety = ttk.LabelFrame(outer, text="扭矩与手动摆位", padding=10)
        safety.grid(row=3, column=0, sticky="new", pady=(0, 10))
        safety.columnconfigure(0, weight=1)
        safety.columnconfigure(1, weight=1)
        self.enable_button = ttk.Button(
            safety, text="上电 / 使能扭矩", command=self.enable_torque
        )
        self.enable_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.disable_button = ttk.Button(
            safety, text="掉电 / 取消使能", command=self.disable_torque
        )
        self.disable_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        ttk.Label(safety, textvariable=self.torque_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Label(
            safety,
            text="保存位置前必须先掉电，再手动摆好整只手。",
            style="Hint.TLabel",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        pose = ttk.LabelFrame(outer, text="位置保存与调用", padding=10)
        pose.grid(row=4, column=0, sticky="new", pady=(0, 10))
        pose.columnconfigure(0, weight=1)
        pose.columnconfigure(1, weight=1)
        self.save_button = ttk.Button(
            pose, text="读取并保存当前位置…", command=self.save_current_pose
        )
        self.save_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.use_button = ttk.Button(
            pose, text="连续移动到保存位置", command=self.use_selected_pose
        )
        self.use_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.pose_combo = ttk.Combobox(
            pose, textvariable=self.pose_var, state="readonly", width=38
        )
        self.pose_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(pose, textvariable=self.position_var).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Label(
            pose,
            text=f"保存目录：{self.positions_dir}",
            style="Hint.TLabel",
            wraplength=760,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        status = ttk.LabelFrame(outer, text="状态", padding=10)
        status.grid(row=5, column=0, sticky="nsew")
        status.columnconfigure(0, weight=1)
        status.rowconfigure(1, weight=1)
        ttk.Label(status, textvariable=self.status_var, wraplength=760).grid(
            row=0, column=0, sticky="w"
        )
        self.log_text = tk.Text(status, height=8, state="disabled", wrap="word")
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

    def log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)
        self.log(message)

    def _run_async(self, action: str, function) -> None:
        if self.operation_thread is not None and self.operation_thread.is_alive():
            self._set_status("另一个操作正在进行，请稍候。")
            return

        def worker() -> None:
            with self.operation_lock:
                try:
                    result = function()
                    self.operation_queue.put(("ok", (action, result)))
                except Exception as exc:  # surface hardware errors in the GUI thread
                    self.operation_queue.put(("error", (action, exc)))

        self.operation_thread = threading.Thread(target=worker, daemon=True)
        self.operation_thread.start()

    def connect(self) -> None:
        try:
            baud = int(self.baud_var.get())
            if baud <= 0:
                raise ValueError("波特率必须为正数")
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc), parent=self.root)
            return

        def operation():
            if self.harness is not None:
                self.harness.shutdown()
            self.SafeDynamixelHarness = load_craft_api(self.args.api_root)
            self.harness = self.SafeDynamixelHarness(
                MOTOR_IDS, self.port_var.get().strip(), baud, mode="live"
            )
            self.harness.__enter__()
            self.harness.preflight_check()
            self.torque_enabled = False
            return True

        self._run_async("connect", operation)

    def enable_torque(self) -> None:
        def operation():
            if self.harness is None:
                raise RuntimeError("请先连接并完成自检")
            # Establish the current pose as the goal before enabling torque,
            # preventing a jump caused by a stale goal position.
            self.harness.write_current_positions_as_goal()
            self.harness.enable_torque()
            self.torque_enabled = True

        self._run_async("enable", operation)

    def disable_torque(self) -> None:
        def operation():
            if self.harness is None:
                raise RuntimeError("当前未连接")
            self.harness.end_live_watchdog()
            self.harness._torque_off_all()
            self.harness._torque_enabled = False
            self.torque_enabled = False

        self._run_async("disable", operation)

    def _read_current(self) -> np.ndarray:
        if self.harness is None:
            raise RuntimeError("请先连接并完成自检")
        positions = np.asarray(self.harness.read_pos(), dtype=np.float64).reshape(-1)
        if positions.size != len(MOTOR_IDS) or not np.all(np.isfinite(positions)):
            raise RuntimeError("未读取到完整的 15 电机位置")
        return positions

    def save_current_pose(self) -> None:
        if self.torque_enabled:
            messagebox.showwarning(
                "请先掉电",
                "保存手动位置前，请先点击“掉电 / 取消使能”，再摆好机械手。",
                parent=self.root,
            )
            return
        name = simpledialog.askstring(
            "保存位置", "输入位置名称（同名文件会覆盖）：", parent=self.root
        )
        if not name or not name.strip():
            return
        safe_name = "".join(ch for ch in name.strip() if ch not in '\\/:*?"<>|')
        if not safe_name:
            messagebox.showerror("名称错误", "位置名称无效。", parent=self.root)
            return

        def operation():
            positions = self._read_current()
            path = self.positions_dir / f"{safe_name}.json"
            write_pose(path, positions)
            return path

        self._run_async("save", operation)

    def use_selected_pose(self) -> None:
        name = self.pose_var.get()
        path = self.pose_files.get(name)
        if path is None:
            self._set_status("请先选择一个已保存位置。")
            return
        if not self.torque_enabled:
            messagebox.showwarning(
                "请先上电",
                "调用保存位置前，请先点击“上电 / 使能扭矩”。",
                parent=self.root,
            )
            return

        def operation():
            _motor_ids, positions = read_pose(path)
            # One complete goal-position command lets the motor's own profile
            # move continuously. No GUI-side step/jog loop is used here.
            self.harness.monitored_position_move(
                dict(zip(MOTOR_IDS, positions.tolist())),
                duration_s=float(self.args.move_duration),
                max_delta_rad=float("inf"),
            )
            return path.stem

        self._run_async("use", operation)

    def refresh_pose_list(self) -> None:
        self.pose_files.clear()
        if self.positions_dir.is_dir():
            for path in sorted(self.positions_dir.glob("*.json")):
                try:
                    read_pose(path)
                except Exception:
                    continue
                self.pose_files[path.stem] = path
        names = list(self.pose_files)
        self.pose_combo["values"] = names
        if self.pose_var.get() not in self.pose_files:
            self.pose_var.set(names[0] if names else "选择已保存位置…")

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.operation_queue.get_nowait()
                action, result = payload
                if kind == "error":
                    self._set_status(f"{action} 失败：{result}")
                    messagebox.showerror("CRAFT Hand 操作失败", str(result), parent=self.root)
                    continue
                if action == "connect":
                    self.torque_var.set("扭矩：已连接，当前为掉电")
                    self._set_status("连接和自检成功；扭矩保持关闭，可以手动摆位。")
                elif action == "enable":
                    self.torque_var.set("扭矩：已使能")
                    self._set_status("扭矩已使能；可以调用保存位置。")
                elif action == "disable":
                    self.torque_var.set("扭矩：已关闭")
                    self._set_status("扭矩已关闭；现在可以手动摆位并保存。")
                elif action == "save":
                    self.refresh_pose_list()
                    self.pose_var.set(result.stem)
                    self._set_status(f"已保存位置：{result.name}")
                elif action == "use":
                    self._set_status(f"已发送连续位置目标：{result}")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _poll_position(self) -> None:
        if (
            self.harness is not None
            and (self.operation_thread is None or not self.operation_thread.is_alive())
        ):
            try:
                values = self._read_current()
                self.position_var.set(
                    "当前位置：" + "  ".join(f"M{i} {v:+.2f}" for i, v in zip(MOTOR_IDS, values))
                )
            except Exception:
                pass
        self.root.after(250, self._poll_position)

    def close(self) -> None:
        try:
            if self.harness is not None:
                self.harness.shutdown()
        except Exception as exc:
            self.log(f"关闭硬件连接时出现异常：{exc}")
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone CRAFT Hand pose GUI")
    parser.add_argument("--port", default=DEFAULT_PORT, help="CRAFT Hand serial port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Dynamixel baudrate")
    parser.add_argument(
        "--api-root",
        type=Path,
        default=default_api_root(),
        help="CRAFT-Hand_API checkout (defaults to the sibling repository)",
    )
    parser.add_argument(
        "--positions-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "trajectories" / "craft_hand_positions",
        help="Directory for named hand-pose JSON files",
    )
    parser.add_argument(
        "--move-duration",
        type=float,
        default=DEFAULT_MOVE_DURATION_S,
        help="Continuous move command duration/watchdog window in seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.move_duration <= 0 or args.move_duration > 2.0:
        raise SystemExit("--move-duration must be between 0 and 2 seconds")
    root = tk.Tk()
    HandPoseGUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
