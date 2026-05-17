"""Tkinter UI for keyboard car control."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("当前 Python 环境缺少 tkinter，无法启动图形界面") from exc

try:
    from .hardware import CarHttpController
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from hardware import CarHttpController


SUPPORTED_HARDWARE = ["小车", "无人机（待实现）", "机械臂（待实现）"]

KEY_TO_SIGNAL: Dict[str, str] = {
    "i": "前进",
    "k": "后退",
    "j": "左转",
    "l": "右转",
    "space": "停止",
}


@dataclass
class ControlConfig:
    host: str
    port: int
    speed: int
    hardware: str
    mode: str


class KeyboardCarControlUI:
    def __init__(self, host: str, port: int, speed: int) -> None:
        self.root = tk.Tk()
        self.root.title("BCI 小车控制中心")
        self.root.geometry("1120x760")
        self.root.minsize(980, 680)

        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.host_var = tk.StringVar(value=host)
        self.port_var = tk.StringVar(value=str(port))
        self.speed_var = tk.StringVar(value=str(speed))
        self.hardware_var = tk.StringVar(value="小车")
        self.mode_var = tk.StringVar(value="键盘")
        self.keyboard_enabled_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="就绪：先完成配置，再启用键盘控制")
        self.detail_var = tk.StringVar(value="当前仅实现小车的键盘控制，脑机模式后续再接入。")

        self._build_ui()
        self._bind_events()
        self._refresh_state()

    def _build_ui(self) -> None:
        self.root.configure(bg="#111827")

        container = ttk.Frame(self.root, padding=18)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x")
        ttk.Label(header, text="BCI 小车控制中心", font=("Microsoft YaHei UI", 20, "bold")).pack(anchor="w")
        ttk.Label(
            header,
            text="鼠标操作 + 键盘控制 | 先完成键盘模式，脑机模式预留入口",
            font=("Microsoft YaHei UI", 10),
        ).pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(container)
        body.pack(fill="both", expand=True, pady=(16, 12))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, padding=14)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = ttk.Frame(body, padding=14)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self._build_configuration_panel(left)
        self._build_control_panel(right)

        footer = ttk.Frame(container)
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status_var, font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        ttk.Label(footer, textvariable=self.detail_var).pack(anchor="w", pady=(2, 0))

        help_box = ttk.LabelFrame(container, text="配置说明与操作流程", padding=12)
        help_box.pack(fill="both", expand=False, pady=(10, 0))
        self.help_text = tk.Text(help_box, height=10, wrap="word", relief="flat", bg="#f8fafc")
        self.help_text.pack(fill="both", expand=True)
        self.help_text.insert(
            "1.0",
            "1. 配置硬件：当前先选“小车”，树莓派地址默认 192.168.149.1，端口默认 5000，速度默认 50。\n"
            "2. 选择控制方式：当前先用“键盘”，脑机模式先保留入口。\n"
            "3. 点击“应用配置”，再点击“启用键盘控制”。\n"
            "4. 使用按钮或按键 i/j/k/l/空格 控制前进、左转、后退、右转和停止。\n"
            "5. 如果硬件服务没启动，请先在树莓派端启动接收 /signal 的服务。\n"
            "6. 关闭窗口前建议点击“停止并停用键盘”。\n\n"
            "说明：当前 UI 已把键盘控制与鼠标按钮集成到一起，脑机控制入口已预留，但尚未接入。",
        )
        self.help_text.configure(state="disabled")

    def _build_configuration_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="硬件与模式配置", padding=12)
        panel.pack(fill="both", expand=True)

        form = ttk.Frame(panel)
        form.pack(fill="x")
        for idx in range(2):
            form.columnconfigure(idx, weight=1)

        ttk.Label(form, text="控制硬件").grid(row=0, column=0, sticky="w")
        self.hardware_combo = ttk.Combobox(form, textvariable=self.hardware_var, values=SUPPORTED_HARDWARE, state="readonly")
        self.hardware_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 12))

        ttk.Label(form, text="控制方式").grid(row=2, column=0, sticky="w")
        mode_frame = ttk.Frame(form)
        mode_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 12))
        ttk.Radiobutton(mode_frame, text="键盘", variable=self.mode_var, value="键盘", command=self._refresh_state).pack(side="left")
        ttk.Radiobutton(mode_frame, text="脑机（待实现）", variable=self.mode_var, value="脑机（待实现）", command=self._refresh_state).pack(side="left", padx=(12, 0))

        ttk.Label(form, text="树莓派地址").grid(row=4, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.host_var).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 12))

        ttk.Label(form, text="端口").grid(row=6, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.port_var).grid(row=7, column=0, sticky="ew", pady=(4, 12))

        ttk.Label(form, text="速度").grid(row=6, column=1, sticky="w")
        ttk.Entry(form, textvariable=self.speed_var).grid(row=7, column=1, sticky="ew", pady=(4, 12), padx=(8, 0))

        self.apply_button = ttk.Button(panel, text="应用配置", command=self.apply_configuration)
        self.apply_button.pack(fill="x", pady=(0, 8))

        self.keyboard_button = ttk.Button(panel, text="启用键盘控制", command=self.toggle_keyboard_control)
        self.keyboard_button.pack(fill="x", pady=(0, 8))

        self.stop_button = ttk.Button(panel, text="停止并停用键盘", command=self.stop_and_disable_keyboard)
        self.stop_button.pack(fill="x")

        note = ttk.Label(
            panel,
            text="当前仅支持：小车 + 键盘。其他硬件和脑机模式将保留入口。",
            wraplength=360,
            justify="left",
        )
        note.pack(anchor="w", pady=(12, 0))

    def _build_control_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="鼠标控制面板", padding=12)
        panel.pack(fill="both", expand=True)

        grid = ttk.Frame(panel)
        grid.pack(fill="both", expand=True)
        for idx in range(3):
            grid.columnconfigure(idx, weight=1)
        for idx in range(3):
            grid.rowconfigure(idx, weight=1)

        self.forward_button = ttk.Button(grid, text="前进 (i)", command=lambda: self.send_signal("前进", "按钮"))
        self.left_button = ttk.Button(grid, text="左转 (j)", command=lambda: self.send_signal("左转", "按钮"))
        self.stop_pad_button = ttk.Button(grid, text="停止 (SPACE)", command=lambda: self.send_signal("停止", "按钮"))
        self.right_button = ttk.Button(grid, text="右转 (l)", command=lambda: self.send_signal("右转", "按钮"))
        self.backward_button = ttk.Button(grid, text="后退 (k)", command=lambda: self.send_signal("后退", "按钮"))

        self.forward_button.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        self.left_button.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.stop_pad_button.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)
        self.right_button.grid(row=1, column=2, sticky="nsew", padx=6, pady=6)
        self.backward_button.grid(row=2, column=1, sticky="nsew", padx=6, pady=6)

        shortcut_box = ttk.LabelFrame(panel, text="快捷说明", padding=10)
        shortcut_box.pack(fill="x", pady=(12, 0))
        ttk.Label(
            shortcut_box,
            text="按键：i 前进 | k 后退 | j 左转 | l 右转 | 空格 停止\n鼠标：点击对应按钮即可发送信号",
            justify="left",
        ).pack(anchor="w")

        log_box = ttk.LabelFrame(panel, text="运行日志", padding=10)
        log_box.pack(fill="both", expand=True, pady=(12, 0))
        log_frame = ttk.Frame(log_box)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _bind_events(self) -> None:
        self.root.bind_all("<KeyPress>", self._on_key_press)
        self.root.bind_all("<KeyRelease>", self._on_key_release)
        self.hardware_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_state())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _read_config(self) -> ControlConfig:
        host = self.host_var.get().strip() or "192.168.149.1"
        port = int(self.port_var.get().strip())
        speed = int(self.speed_var.get().strip())
        return ControlConfig(
            host=host,
            port=port,
            speed=speed,
            hardware=self.hardware_var.get(),
            mode=self.mode_var.get(),
        )

    def _validate_supported(self, config: ControlConfig) -> bool:
        if config.hardware != "小车":
            self._set_status("当前仅支持小车控制", f"已选择 {config.hardware}，请先切回小车。")
            return False
        if config.mode != "键盘":
            self._set_status("当前仅支持键盘控制", "脑机控制入口已预留，暂未接入。")
            return False
        return True

    def _set_status(self, status: str, detail: str) -> None:
        self.status_var.set(status)
        self.detail_var.set(detail)

    def _log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _refresh_state(self) -> None:
        try:
            config = self._read_config()
        except Exception:
            self._set_status("配置未通过验证", "请检查端口和速度是否为整数。")
            self._set_buttons_enabled(False)
            self.keyboard_enabled_var.set(False)
            self.keyboard_button.configure(text="启用键盘控制")
            return

        supported = self._validate_supported(config)
        self._set_buttons_enabled(supported)
        if supported:
            if self.keyboard_enabled_var.get():
                self._set_status("键盘控制已启用", f"目标 {config.host}:{config.port}，速度 {config.speed}")
            else:
                self._set_status("配置已就绪", f"目标 {config.host}:{config.port}，速度 {config.speed}")

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in (
            self.keyboard_button,
            self.stop_button,
            self.forward_button,
            self.left_button,
            self.stop_pad_button,
            self.right_button,
            self.backward_button,
        ):
            button.configure(state=state)

    def apply_configuration(self) -> None:
        try:
            config = self._read_config()
        except Exception as exc:
            messagebox.showerror("配置错误", f"端口和速度必须是整数。\n{exc}")
            self._log(f"[配置失败] {exc}")
            return

        if not self._validate_supported(config):
            self._log(f"[配置] 硬件={config.hardware}，模式={config.mode}，暂未支持")
            return

        self._log(f"[配置] 硬件={config.hardware}，模式={config.mode}，目标={config.host}:{config.port}，速度={config.speed}")
        self._set_status("配置已应用", f"目标 {config.host}:{config.port}，速度 {config.speed}")
        self._refresh_state()

    def toggle_keyboard_control(self) -> None:
        if self.mode_var.get() != "键盘":
            self._log("[提示] 脑机模式尚未接入，请先选择键盘模式")
            return
        if self.hardware_var.get() != "小车":
            self._log("[提示] 当前仅支持小车控制")
            return

        self.keyboard_enabled_var.set(not self.keyboard_enabled_var.get())
        if self.keyboard_enabled_var.get():
            self.keyboard_button.configure(text="停用键盘控制")
            self._set_status("键盘控制已启用", "现在可以使用 i/j/k/l/空格 或鼠标按钮控制小车。")
            self._log("[键盘] 已启用")
            self.root.focus_force()
        else:
            self.keyboard_button.configure(text="启用键盘控制")
            self._set_status("键盘控制已停用", "如需继续控制，请再次启用键盘控制。")
            self._log("[键盘] 已停用")

    def stop_and_disable_keyboard(self) -> None:
        self.send_signal("停止", "按钮")
        if self.keyboard_enabled_var.get():
            self.keyboard_enabled_var.set(False)
            self.keyboard_button.configure(text="启用键盘控制")
        self._set_status("已停止", "控制已停止，键盘监听已关闭。")
        self._log("[控制] 停止并关闭键盘监听")

    def _send_signal_worker(self, config: ControlConfig, signal: str, source: str) -> None:
        try:
            controller = CarHttpController(host=config.host, port=config.port, speed=config.speed)
            result = controller.send_signal(signal)
            self.root.after(0, lambda: self._on_send_success(config, signal, source, result))
        except Exception as exc:
            self.root.after(0, lambda: self._on_send_error(config, signal, source, exc))

    def _on_send_success(self, config: ControlConfig, signal: str, source: str, result: object) -> None:
        self._set_status("发送成功", f"{source} -> {signal} | {config.host}:{config.port} | 速度 {config.speed}")
        self._log(f"[成功] {source} -> {signal} | 返回: {result}")

    def _on_send_error(self, config: ControlConfig, signal: str, source: str, exc: Exception) -> None:
        message = f"{source} -> {signal} 发送失败: {exc}"
        self._set_status("发送失败", message)
        self._log(f"[失败] {message}")

    def send_signal(self, signal: str, source: str) -> None:
        try:
            config = self._read_config()
        except Exception as exc:
            self._log(f"[失败] 配置错误: {exc}")
            messagebox.showerror("配置错误", f"端口和速度必须是整数。\n{exc}")
            return

        if not self._validate_supported(config):
            self._log(f"[忽略] {source} -> {signal}，当前模式暂未支持")
            return

        thread = threading.Thread(target=self._send_signal_worker, args=(config, signal, source), daemon=True)
        thread.start()

    def _on_key_press(self, event: tk.Event) -> None:
        if not self.keyboard_enabled_var.get():
            return
        keysym = (event.keysym or "").lower()
        signal = KEY_TO_SIGNAL.get(keysym)
        if signal:
            self.send_signal(signal, "键盘")

    def _on_key_release(self, event: tk.Event) -> None:
        if not self.keyboard_enabled_var.get():
            return
        keysym = (event.keysym or "").lower()
        if keysym in {"i", "j", "k", "l"}:
            self.send_signal("停止", "键盘释放")

    def on_close(self) -> None:
        try:
            self.send_signal("停止", "关闭窗口")
        finally:
            self.root.destroy()

    def run(self) -> None:
        self._log("[启动] UI 已启动")
        self.root.mainloop()


def launch_control_ui(host: str = "192.168.149.1", port: int = 5000, speed: int = 50) -> None:
    app = KeyboardCarControlUI(host=host, port=port, speed=speed)
    app.run()