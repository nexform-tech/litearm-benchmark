"""界面入口 —— ``litearm-bench-gui`` / ``litearm-bench gui``。

⚠ 这个界面**只是同一套装配路径的另一个前端**。它没有自己的采集、分析或安全判断；
挑动作表 → `session.run_actions` → 显示结果，仅此三件。
"""

from __future__ import annotations

import sys
import traceback


def _install_excepthook() -> None:
    """把未捕获异常**记下来**，而不是让进程 abort。

    ⚠⚠ PyQt5 默认在**槽里**抛出的未捕获异常会直接 `qFatal` ⇒ **整个界面核心转储**。
    一个笔误（比如对 `None` 做算术）就会让操作者看到"窗口突然没了"，
    而那正是最该看到原因的时刻。⇒ 换成打日志。

    唯一一处**不**吞的：`KeyboardInterrupt`（照常退出）。
    """
    def hook(etype, value, tb):
        if issubclass(etype, KeyboardInterrupt):
            sys.__excepthook__(etype, value, tb)
            return
        sys.stderr.write("⚠ 界面里未捕获的异常（已记录，进程继续）：\n")
        traceback.print_exception(etype, value, tb, file=sys.stderr)
    sys.excepthook = hook


def main(argv=None) -> int:
    from PyQt5.QtWidgets import QApplication

    from .main_window import MainWindow

    _install_excepthook()
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("litearm-bench")
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
