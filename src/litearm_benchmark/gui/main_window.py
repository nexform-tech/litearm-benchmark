"""主窗口 —— 档位 / 单项 / 历史 / 自检 四个页，右侧共用一块成绩单。

⚠ **界面不做任何自己的判断**。它能做的只有三件：挑动作表、调 `run_actions`、显示结果。
安全闸在底层（`safety.require` + `Executor._gate`），界面**没有**绕过的路径；
`SafetyAbort` 会原样显示在日志里，而不是被吞掉。

⚠ 跑之前一律要**显式确认**（等价于命令行的 `--go`），且确认框里列出**将要跑什么**。
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from .. import __version__
from ..replay import compare, rescore
from ..suites import singles as _singles
from ..suites import tiers as _tiers
from .bridge import GuiBridge, PromptBox
from .widgets import CardView, DeviceBar, LogPane
from .worker import Worker

DOCS = "docs"


@dataclass(frozen=True)
class PortInfo:
    """一条串口 —— 设备路径、给人看的标签、以及"它是不是咱家的臂"。"""
    device: str
    label: str
    is_arm: bool
    is_usb: bool


def _list_ports() -> List[PortInfo]:
    """列出串口，并**认出自家那台臂**，排到最前、打上 ★。

    ⚠ 认设备一律走 `litearm_core.find_cdc_port()` —— **公开面**。本工程**不复制**那对
    VID:PID（它是 `litearm_core/transport.py` 的私有常数）：复制一份的话，固件那边
    改了 PID，这里会**继续按老 ID 认**，症状是"界面说没识别到，可臂明明插着"。
    家族规矩：**一律按 VID:PID 认设备**，`ttyACM0/1` 的编号会互换（本机 `ttyACM1`
    就是台 ATK CMSIS-DAP 调试器，不是臂）。

    ⚠ 早先这里写的是 `if p not in out: out.insert(0, p)` —— **恒假**：pyserial 自己就会
    列出 CDC 口（本机第 32 位，前面 31 个全是 `/dev/ttyS*`），于是"插到最前"被自己的
    守卫挡掉，下拉默认选中 `/dev/ttyS4`。

    ⚠ 非 USB 串口（`/dev/ttyS*`）**留在这里**、由界面按需过滤，不在本函数里消失 ——
    取数据的函数不该替调用方决定"什么算噪声"。
    """
    arm: Optional[str] = None
    try:
        from litearm_core import find_cdc_port
        arm = find_cdc_port()
    except Exception:                                     # noqa: BLE001
        pass                                                  # 认不出就只是没有 ★

    infos: List[PortInfo] = []
    try:
        import serial.tools.list_ports as lp
        for p in lp.comports():
            is_arm = arm is not None and p.device == arm
            bits = ["LiteArm"] if is_arm else []
            if p.vid is not None:
                bits.append(f"{p.vid:04x}:{p.pid:04x}")
            if p.product:
                bits.append(p.product)
            label = ("★ " if is_arm else "") + p.device \
                + ("　" + "　".join(bits) if bits else "")
            infos.append(PortInfo(p.device, label, is_arm, p.vid is not None))
    except Exception:                                     # noqa: BLE001
        pass

    # 认出来了、pyserial 却没列（罕见）⇒ 补一条，否则界面上**根本选不到**它
    if arm and not any(i.device == arm for i in infos):
        infos.append(PortInfo(arm, f"★ {arm}　LiteArm", True, True))

    infos.sort(key=lambda i: not i.is_arm)                # 稳定排序 ⇒ 臂第一，其余保原序
    return infos


class ConfirmRun(QDialog):
    """跑之前的确认 —— 等价于命令行的 `--go`。**列出将要跑什么**。"""

    def __init__(self, title: str, lines: List[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("确认动臂")
        lay = QVBoxLayout(self)
        head = QLabel(f"<b>{title}</b>")
        lay.addWidget(head)
        body = QPlainTextEdit("\n".join(lines))
        body.setReadOnly(True)
        body.setMinimumWidth(560)
        lay.addWidget(body)
        warn = QLabel("⚠ <b>这会真实驱动机械臂。</b>确认台面清空、无人处于运动范围内。")
        warn.setWordWrap(True)
        lay.addWidget(warn)
        bb = QDialogButtonBox()
        self._go = bb.addButton("开始（会动机器人）", QDialogButtonBox.AcceptRole)
        bb.addButton("取消", QDialogButtonBox.RejectRole)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._go.setDefault(False)

    @staticmethod
    def ask(title: str, lines: List[str], parent=None) -> bool:
        return ConfirmRun(title, lines, parent).exec_() == QDialog.Accepted


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"LiteArm 基准测试台　v{__version__}")
        self.resize(1280, 860)

        self.bridge = GuiBridge()
        self.worker = Worker(self.bridge)
        self._card = None
        self._history: List[str] = []

        self._build()
        self._wire()
        self.worker.start()
        self._refresh_ports()
        self._refresh_history()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        self.dev = DeviceBar()
        outer.addWidget(self.dev)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("端口"))
        self.port = QComboBox()
        self.port.setMinimumWidth(320)
        bar.addWidget(self.port)
        self.btn_refresh = QPushButton("刷新")
        self.btn_conn = QPushButton("连接")
        self.btn_conn.setCheckable(True)
        bar.addWidget(self.btn_refresh)
        bar.addWidget(self.btn_conn)
        #: ⚠ 默认**隐藏非 USB 串口**：`/dev/ttyS*` 是主板串口，物理上不可能是这台臂，
        #:   本机 34 个里 31 个都是它们 ⇒ 藏起来，别让操作者在噪声里找。这是**过滤**、
        #:   不是删除：勾上就回来，认设备的判据本身也从不依赖它。
        self.chk_all = QCheckBox("显示非 USB 串口")
        bar.addWidget(self.chk_all)
        bar.addStretch(1)
        self.lbl_note = QLabel("")
        bar.addWidget(self.lbl_note)
        outer.addLayout(bar)

        split_v = QSplitter(Qt.Vertical)
        split_h = QSplitter(Qt.Horizontal)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._page_tiers(), "档位")
        self.tabs.addTab(self._page_singles(), "单项")
        self.tabs.addTab(self._page_history(), "历史")
        self.tabs.addTab(self._page_doctor(), "自检")
        split_h.addWidget(self.tabs)

        self.card_view = CardView()
        split_h.addWidget(self.card_view)
        split_h.setStretchFactor(1, 3)

        split_v.addWidget(split_h)
        self.log = LogPane()
        split_v.addWidget(self.log)
        split_v.setStretchFactor(0, 4)
        split_v.setStretchFactor(1, 1)
        outer.addWidget(split_v)

    def _page_tiers(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lst_tiers = QListWidget()
        for key, (acts, budget) in _tiers.TIERS.items():
            wall = _tiers.MEASURED_WALL_S.get(key)
            secs = "需人工" if wall is None else f"实测 ~{wall/60:.0f} 分" if wall >= 90 \
                else f"实测 ~{wall:.0f} 秒"
            manual = any(a.is_manual for a in acts)
            it = QListWidgetItem(f"{key}　—　{len(acts)} 条动作　{secs}"
                                 + ("　（含人工步骤）" if manual else ""))
            it.setData(Qt.UserRole, key)
            self.lst_tiers.addItem(it)
        self.lst_tiers.setCurrentRow(0)
        lay.addWidget(self.lst_tiers)
        self.desc_tier = QLabel("")
        self.desc_tier.setWordWrap(True)
        lay.addWidget(self.desc_tier)
        self.btn_run_tier = QPushButton("运行这个档位…")
        lay.addWidget(self.btn_run_tier)
        self.lst_tiers.currentRowChanged.connect(self._on_tier_changed)
        self._on_tier_changed(0)
        return w

    def _page_singles(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lst_singles = QListWidget()
        for it in _singles.SINGLES.values():
            need = f"　需 {'/'.join(it.needs)}" if it.needs else ""
            row = QListWidgetItem(f"{it.title}　(~{it.seconds:.0f}s{need})")
            row.setData(Qt.UserRole, it.key)
            self.lst_singles.addItem(row)
        self.lst_singles.setCurrentRow(0)
        lay.addWidget(self.lst_singles)
        self.desc_single = QLabel("")
        self.desc_single.setWordWrap(True)
        lay.addWidget(self.desc_single)
        self.btn_run_single = QPushButton("运行这个单项…")
        lay.addWidget(self.btn_run_single)
        self.lst_singles.currentRowChanged.connect(self._on_single_changed)
        self._on_single_changed(0)
        return w

    def _page_history(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lst_runs = QListWidget()
        self.lst_runs.setSelectionMode(QAbstractItemView.ExtendedSelection)
        lay.addWidget(self.lst_runs)
        row = QHBoxLayout()
        self.btn_reload = QPushButton("重载历史")
        self.btn_rescore = QPushButton("重算选中（改权重/阈值后）")
        self.btn_compare = QPushButton("对比选中两轮")
        for b in (self.btn_reload, self.btn_rescore, self.btn_compare):
            row.addWidget(b)
        lay.addLayout(row)
        return w

    def _page_doctor(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.txt_doctor = QPlainTextEdit()
        self.txt_doctor.setReadOnly(True)
        self.btn_doctor = QPushButton("跑口径自检（不动机器人）")
        lay.addWidget(self.btn_doctor)
        lay.addWidget(self.txt_doctor)
        return w

    # ------------------------------------------------------------------ #
    def _wire(self) -> None:
        b = self.bridge
        b.sig_log.connect(self.log.append_line)
        b.sig_error.connect(self._on_error)
        b.sig_device.connect(self.dev.set_status)
        b.sig_run_started.connect(self._on_run_started)
        b.sig_run_finished.connect(self._on_run_finished)
        b.sig_task.connect(self._on_task)
        b.sig_prompt.connect(self._on_prompt)

        self.btn_refresh.clicked.connect(self._refresh_ports)
        self.chk_all.toggled.connect(self._refresh_ports)
        self.btn_conn.clicked.connect(self._toggle_connect)
        self.btn_run_tier.clicked.connect(self._run_tier)
        self.btn_run_single.clicked.connect(self._run_single)
        self.btn_reload.clicked.connect(self._refresh_history)
        self.btn_rescore.clicked.connect(self._rescore)
        self.btn_compare.clicked.connect(self._compare)
        self.btn_doctor.clicked.connect(self._doctor)

    def closeEvent(self, ev) -> None:                     # noqa: N802 - Qt
        if self.worker.isRunning():
            self.log.append_line("⏹ 关闭：正在收尾工作线程（臂保持使能，不失能）")
        self.worker.shutdown()
        super().closeEvent(ev)

    # ------------------------------------------------------------------ #
    def _refresh_ports(self) -> None:
        # ⚠ 记住的是**设备路径**（`currentData`）而不是显示文字：标签里带 ★ 和
        #   VID:PID，一旦那台臂插拔后 ID 变了，按文字回选就会**静默丢选**。
        cur = self.port.currentData()
        self.port.clear()
        infos = _list_ports()
        shown = infos if self.chk_all.isChecked() else \
            [i for i in infos if i.is_arm or i.is_usb]
        for i in shown:
            self.port.addItem(i.label, i.device)
        if cur:
            idx = self.port.findData(cur)
            if idx >= 0:
                self.port.setCurrentIndex(idx)

        arm = next((i for i in infos if i.is_arm), None)
        if arm is not None:
            self.lbl_note.setText(f"已识别 LiteArm：<b>{arm.device}</b>")
        elif not infos:
            self.lbl_note.setText("没有发现串口")
        else:
            self.lbl_note.setText(
                f"<b>未识别到 LiteArm</b>（已列出 {len(shown)} 个串口，可手动选）")

    def _toggle_connect(self, on: bool) -> None:
        if on:
            self.worker.op_set_port(self.port.currentData() or None)
            self.btn_conn.setText("断开")
        else:
            self.worker.op_set_port(None)
            self.btn_conn.setText("连接")

    # ---- 交互 ----
    def _on_tier_changed(self, row: int) -> None:
        if row < 0 or row >= self.lst_tiers.count():
            return
        key = self.lst_tiers.item(row).data(Qt.UserRole)
        acts = _tiers.actions_of(key)
        budget = _tiers.budget_of(key)
        wall = _tiers.MEASURED_WALL_S.get(key)
        tgt = sorted({a.target for a in acts if a.target} |
                     {a.via for a in acts if a.via})
        # ⚠ `manual_load` 的预算是 `None`（时长取决于操作者）—— 别对它做算术。
        #   （曾经写成 `{budget/60:.0f}`，`None/60` 抛异常，而 **Qt 槽里抛异常会 abort**，
        #     界面直接核心转储。任何进槽的代码都要假设自己可能拿到 None。）
        budget_txt = "不设截断死线" if budget is None else f"截断死线 {budget/60:.0f} 分"
        self.desc_tier.setText(
            f"<b>{key}</b>　{len(acts)} 条动作　"
            f"实测约 {'需人工' if wall is None else f'{wall:.0f} 秒'}"
            f"（{budget_txt}）<br>"
            f"位形：{', '.join(tgt) or '—'}"
            + ("　<b>⚠ 含人工步骤</b>（装配重/写前馈）" if any(a.is_manual for a in acts)
               else ""))

    def _on_single_changed(self, row: int) -> None:
        if row < 0 or row >= self.lst_singles.count():
            return
        key = self.lst_singles.item(row).data(Qt.UserRole)
        it = _singles.SINGLES[key]
        self.desc_single.setText(f"<b>{it.title}</b><br>{it.desc}")

    # ---- 运行 ----
    def _confirm(self, title: str, acts) -> bool:
        lines = ["将要执行："]
        for a in acts:
            t = f" → {a.target}" if a.target else ""
            v = f"（每轮先经 {a.via}）" if a.via else ""
            r = f" ×{a.reps}" if a.reps > 1 else ""
            s = f" @speed {a.speed}" if a.kind.startswith("move") else ""
            hint = f"　【需人工】{a.manual_hint}" if a.is_manual else ""
            lines.append(f"  · {a.kind}　{a.label}{t}{v}{r}{s}{hint}")
        lines.append("")
        lines.append("每一步都会过安全闸（限位 + 锁存 + 温度），闸拒绝时不会动。")
        return ConfirmRun.ask(title, lines, self)

    def _run_tier(self) -> None:
        row = self.lst_tiers.currentRow()
        if row < 0:
            return
        key = self.lst_tiers.item(row).data(Qt.UserRole)
        acts = _tiers.actions_of(key)
        if not self._confirm(f"档位 {key}", acts):
            return
        outdir = os.path.join(DOCS, f"bench_{time.strftime('%Y%m%d_%H%M')}_{key}")
        self.worker.op_run(actions=acts, label=key, outdir=outdir)

    def _run_single(self) -> None:
        row = self.lst_singles.currentRow()
        if row < 0:
            return
        key = self.lst_singles.item(row).data(Qt.UserRole)
        it = _singles.SINGLES[key]
        if not self._confirm(f"单项 {it.title}", list(it.actions)):
            return
        outdir = os.path.join(DOCS, f"bench_{time.strftime('%Y%m%d_%H%M')}_{key}")
        self.worker.op_run(actions=list(it.actions), label=key, outdir=outdir)

    # ---- 回执 ----
    def _on_run_started(self, label: str) -> None:
        self.log.append_line(f"▶ 开始 {label}（工作线程已让出串口，基准会自己开连接）")
        self.btn_run_tier.setEnabled(False)
        self.btn_run_single.setEnabled(False)

    def _on_run_finished(self, res) -> None:
        self.btn_run_tier.setEnabled(True)
        self.btn_run_single.setEnabled(True)
        self._card = res.card if res is not None else None
        if res is not None:
            self.card_view.set_card(res.card, raw=res.raw,
                                    title=f"{res.card.identity.get('firmware', '')}　"
                                          f"{getattr(res, 'outdir', '') or ''}")
            if res.outdir:
                self.log.append_line(f"📄 落盘：{res.outdir}")
        else:
            self.card_view.set_card(None, title="本轮没有结果")
        self._refresh_history()

    def _on_error(self, msg: str) -> None:
        self.log.append_line(msg)
        if msg.startswith("⛔"):
            self.btn_run_tier.setEnabled(True)
            self.btn_run_single.setEnabled(True)

    def _on_prompt(self, box: PromptBox) -> None:
        d = QMessageBox(self)
        d.setWindowTitle("需人工")
        d.setIcon(QMessageBox.Warning)
        d.setText(box.hint)
        d.setInformativeText("完成并确认设备已恢复后点「已完成」；点「跳过」则不执行这一步。")
        done = d.addButton("已完成", QMessageBox.AcceptRole)
        d.addButton("跳过", QMessageBox.RejectRole)
        d.exec_()
        box.answer = "ok" if d.clickedButton() is done else "skip"
        box.event.set()

    def _on_task(self, name: str, payload) -> None:
        if name == "doctor":
            self.txt_doctor.setPlainText(str(payload))
        elif name == "identity":
            self.log.append_line("设备身份：" + json.dumps(payload, ensure_ascii=False)[:400])
        elif name == "rescore":
            self.log.append_line(f"重算完成：{payload}")
            self._refresh_history()
        elif name == "compare":
            self.log.append_line("—— 对比 ——")
            for ln in str(payload).splitlines():
                self.log.append_line(ln)
        else:
            self.log.append_line(f"[{name}] {payload}")

    # ---- 历史 ----
    def _refresh_history(self) -> None:
        runs = sorted(glob.glob(os.path.join(DOCS, "bench_*")),
                      key=os.path.getmtime, reverse=True)
        self._history = runs
        self.lst_runs.clear()
        for d in runs[:200]:
            total = grade = "—"
            try:
                raw = json.load(open(os.path.join(d, "raw.json"), encoding="utf-8"))
                meta = raw.get("meta") or {}
                total = "仿真" if meta.get("simulated") else ""
            except Exception:                             # noqa: BLE001
                pass
            row = QListWidgetItem(f"{os.path.basename(d)}　{total}")
            row.setData(Qt.UserRole, d)
            self.lst_runs.addItem(row)

    def _selected_runs(self) -> List[str]:
        return [i.data(Qt.UserRole) for i in self.lst_runs.selectedItems()]

    def _rescore(self) -> None:
        dirs = self._selected_runs()
        if not dirs:
            return
        def fn():
            out = []
            for d in dirs:
                c = rescore(d)
                out.append(f"{os.path.basename(d)}："
                           + ("仿真不出总分" if c.total is None else f"{c.total:.1f} / {c.grade}"))
            return "；".join(out)
        self.worker.op_task("rescore", fn)

    def _compare(self) -> None:
        dirs = self._selected_runs()
        if len(dirs) != 2:
            QMessageBox.information(self, "对比", "请恰好选中两轮（Ctrl 多选）。")
            return
        a, b = sorted(dirs, key=os.path.getmtime)
        self.worker.op_task("compare", lambda: compare(a, b))

    def _doctor(self) -> None:
        def fn() -> str:
            from ..instruments import describe
            from ..metrics.registry import audit_curves, unresolved_placeholders, ALL_METRICS, DIMENSION_ORDER
            bad = audit_curves()
            ph = unresolved_placeholders()
            out = [f"litearm-benchmark {__version__}",
                   f"指标 {len(ALL_METRICS)} 项，维度 {len(DIMENSION_ORDER)} 个",
                   "曲线自检：" + ("✅ 全部合规" if not bad else f"⛔ {len(bad)} 处问题"),
                   *[f"   {b}" for b in bad],
                   f"", f"仍靠**占位**阈值的指标（{len(ph)} 项，分数只有相对意义）：",
                   *[f"   - {k}" for k in ph],
                   "", "档位实测挂钟：" + "、".join(
                       f"{k}={'需人工' if v is None else f'{v:.0f}s'}"
                       for k, v in _tiers.MEASURED_WALL_S.items()),
                   "", "## 仪表边界", describe()]
            return "\n".join(out)
        self.worker.op_task("doctor", fn)
