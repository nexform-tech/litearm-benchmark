"""GuiBridge —— 跨线程 Qt 信号桥。

约定照本家族 `litearm-tool-stm32` 的 `litetool/gui/bridge.py`：
SDK 所有者线程 → `emit` → 主线程 slot（`AutoConnection` 自动转 `QueuedConnection`）。
于是**任何**后台任务的结果都在 GUI 线程上被处理，页面不必自己关心线程边界。
"""

from __future__ import annotations

from PyQt5.QtCore import QObject, pyqtSignal


class PromptBox:
    """人工步骤的**跨线程应答盒**。

    工作线程调到 `_manual` 时会**阻塞**在这里（操作者要动手装/拆配重）；
    主线程弹出对话框，把答案写进 `answer` 后 `event.set()` 放行。

    ⚠ 反向依赖（工作线程直接弹 Qt 对话框）是**不可以**的 —— Qt 控件只能在 GUI 线程里碰。
    """

    __slots__ = ("hint", "answer", "event")

    def __init__(self, hint: str, event) -> None:
        self.hint = hint
        self.answer = "skip"          # 默认跳过：操作者不应答时**不做**危险动作
        self.event = event


class GuiBridge(QObject):
    sig_log = pyqtSignal(str)                 # 一行日志
    sig_error = pyqtSignal(str)               # 一个失败说明
    sig_device = pyqtSignal(object)           # DeviceStatus 快照
    sig_run_started = pyqtSignal(str)         # label
    sig_run_finished = pyqtSignal(object)     # RunResult | None
    sig_task = pyqtSignal(str, object)        # (任务名, 载荷) —— 一次性任务的回执
    sig_prompt = pyqtSignal(object)           # PromptBox —— 人工步骤
