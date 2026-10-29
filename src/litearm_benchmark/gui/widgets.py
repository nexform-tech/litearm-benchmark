"""界面控件 —— 设备栏 / 成绩单 / 日志面板。

⚠ 成绩单这一块**必须把三条硬规矩显示出来**，否则界面会比命令行更容易骗人：

1. **未实测 ≠ 0 分** —— 未实测的指标要显式标出来并写明理由，不能让它看起来像"得了低分"；
2. **不可信 ≠ 高分** —— 自证量不达标的指标要说清"这数不能用"，而不是照常显示；
3. **仿真绝不出总分** —— 桩数据要在最显眼处挂红条。
"""

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QFrame, QHBoxLayout, QHeaderView, QLabel, QPlainTextEdit,
    QSizePolicy, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
    QWidget,
)

from .worker import TEMP_ALARM_C, DeviceStatus

_GREEN = QColor("#1b7f3b")
_RED = QColor("#b3261e")
_AMBER = QColor("#8a6100")
_GREY = QColor("#5f6368")


def _item(text: str, *, color: Optional[QColor] = None, bold: bool = False,
          tip: str = "") -> QTableWidgetItem:
    it = QTableWidgetItem(str(text))
    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    if color is not None:
        it.setForeground(color)
    if bold:
        f = it.font()
        f.setBold(True)
        it.setFont(f)
    if tip:
        it.setToolTip(tip)
    return it


class LogPane(QPlainTextEdit):
    """只读日志。**上限行数**，免得 6 小时浸泡把内存吃光。"""

    MAX_BLOCKS = 5000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_BLOCKS)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        f = QFont("monospace")
        f.setStyleHint(QFont.TypeWriter)
        self.setFont(f)

    def append_line(self, msg: str) -> None:
        self.appendPlainText(str(msg))


class DeviceBar(QFrame):
    """设备栏 —— 在线/固件/使能/锁存/温度/裕度。**取不到就显示「未知」，不填默认值。**"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        self._dot = QLabel("●")
        self._dot.setStyleSheet(f"color: {_GREY.name()}; font-size: 18px;")
        self._main = QLabel("未连接")
        self._main.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._main.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay.addWidget(self._dot)
        lay.addWidget(self._main)
        self.setFrameStyle(QFrame.StyledPanel)

    def set_status(self, s: DeviceStatus) -> None:
        if not s.online:
            self._dot.setStyleSheet(f"color: {_GREY.name()}; font-size: 18px;")
            self._main.setText(f"未连接　{s.port}　{s.why}")
            return
        hot = s.t_max >= TEMP_ALARM_C or s.t_coil_max >= TEMP_ALARM_C
        bad = bool(s.joint_fault) or bool(s.faults) or hot
        self._dot.setStyleSheet(
            f"color: {(_RED if bad else _GREEN).name()}; font-size: 18px;")
        bits = [
            f"<b>{s.port}</b>",
            f"固件 {s.firmware or '—'}",
            f"{s.n_joints} 轴",
            f"使能 {'是' if s.enabled else '<b>否</b>'}",
            f"模式 {s.mode or '—'}",
            f"温度 {s.t_max:.0f}/{s.t_coil_max:.0f} °C"
            + (f" <b>⚠ 逼近 {TEMP_ALARM_C:.0f} 停跑线</b>" if hot else ""),
            f"最小距锁存线裕度 <b>{s.margin:+.4f}</b> rad @J{s.tightest}",
        ]
        if s.joint_fault:
            bits.append(f"<b style='color:{_RED.name()}'>joint_fault=0x{s.joint_fault:04X}</b>")
        if s.faults:
            bits.append(f"<b style='color:{_RED.name()}'>安全前置：{'；'.join(s.faults)}</b>")
        for w in s.warnings:
            bits.append(f"<span style='color:{_AMBER.name()}'>⚠ {w}</span>")
        self._main.setText("　|　".join(bits))


class CardView(QWidget):
    """成绩单：总分条 + 维度表 + 指标明细 + 未实测清单。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._banner = QLabel("尚未运行")
        self._banner.setWordWrap(True)
        self._banner.setStyleSheet("padding:6px; background:#f1f3f4;")
        lay.addWidget(self._banner)

        self._tabs = QTabWidget()
        self._dims = QTableWidget(0, 4)
        self._dims.setHorizontalHeaderLabels(["维度", "权重", "得分", "已计分"])
        self._dims.verticalHeader().setVisible(False)
        self._dims.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._dims.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._dims.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._dims.itemSelectionChanged.connect(self._on_dim)

        self._metrics = QTableWidget(0, 6)
        self._metrics.setHorizontalHeaderLabels(
            ["指标", "原始值", "分", "状态", "单位", "证据 / 说明"])
        self._metrics.verticalHeader().setVisible(False)
        self._metrics.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._metrics.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)

        self._skipped = QPlainTextEdit()
        self._skipped.setReadOnly(True)
        self._skipped.setLineWrapMode(QPlainTextEdit.WidgetWidth)

        self._tabs.addTab(self._dims, "维度")
        self._tabs.addTab(self._metrics, "指标明细")
        self._tabs.addTab(self._skipped, "未实测 / 不可信")
        lay.addWidget(self._tabs)
        self._card = None

    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        self._card = None
        self._banner.setText("尚未运行")
        self._banner.setStyleSheet("padding:6px; background:#f1f3f4;")
        for t in (self._dims, self._metrics):
            t.setRowCount(0)
        self._skipped.clear()

    def set_card(self, card, *, raw=None, title: str = "") -> None:
        self._card = card
        if card is None:
            self.clear()
            self._banner.setText(f"{title}　—　没有结果")
            return

        # ---- 总分条：三条硬规矩都在这里体现 ----
        if card.simulated:
            self._banner.setText(
                "⚠ <b>仿真 / 桩数据</b> —— 桩不建模运动 ⇒ 抖动 0 与 τ=0 都会伪装成满分。"
                "<b>本工程绝不给仿真出总分</b>，所有指标一律标为不可信。")
            self._banner.setStyleSheet(
                f"padding:6px; background:#fdecea; color:{_RED.name()};")
        elif card.total is None:
            self._banner.setText(
                "总分：<b>无可计分维度</b> —— 不产生总分（<b>未实测 ≠ 0 分</b>）。"
                "原因见「未实测 / 不可信」页。")
            self._banner.setStyleSheet(
                f"padding:6px; background:#fff8e1; color:{_AMBER.name()};")
        else:
            self._banner.setText(
                f"{title}　总分 <b style='font-size:18px'>{card.total:.1f}</b> / 100"
                f"　等级 <b>{card.grade}</b>"
                f"　（{len(card.scored_dimensions)} 个已测维度加权；未测维度**不参与**、"
                f"也不从 0 分起算）")
            self._banner.setStyleSheet("padding:6px; background:#e8f0fe;")

        # ---- 维度表 ----
        self._dims.setRowCount(len(card.dimensions))
        for r, d in enumerate(card.dimensions):
            sc = d.score
            col = _GREY if sc is None else (
                _GREEN if sc >= 80 else (_AMBER if sc >= 60 else _RED))
            self._dims.setItem(r, 0, _item(d.name))
            self._dims.setItem(r, 1, _item(f"{d.weight:g}"))
            self._dims.setItem(r, 2, _item("—" if sc is None else f"{sc:.1f}", color=col,
                                           bold=True))
            self._dims.setItem(r, 3, _item(d.measured_ratio))
        self._dims.resizeColumnsToContents()
        self._dims.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        if card.dimensions:
            self._dims.selectRow(0)
            self._show_dim(0)
        self._fill_skipped(card)

    def _on_dim(self) -> None:
        rows = self._dims.selectionModel().selectedRows()
        if rows:
            self._show_dim(rows[0].row())

    def _show_dim(self, row: int) -> None:
        if self._card is None or row >= len(self._card.dimensions):
            return
        ms = self._card.dimensions[row].metrics
        self._metrics.setRowCount(len(ms))
        for r, m in enumerate(ms):
            if m.score is None:
                state, col = "未实测", _GREY
            elif not m.trusted:
                state, col = "不可信", _RED
            elif not m.counted:
                state, col = "不计分", _AMBER
            else:
                state, col = "计分", _GREEN
            raw = "—" if m.raw is None else (
                f"{m.raw:.4g}" if isinstance(m.raw, float) else str(m.raw))
            self._metrics.setItem(r, 0, _item(m.spec.key, tip=m.spec.note or ""))
            self._metrics.setItem(r, 1, _item(raw))
            self._metrics.setItem(r, 2, _item("—" if m.score is None else f"{m.score:.0f}",
                                              color=col, bold=True))
            self._metrics.setItem(r, 3, _item(state, color=col))
            self._metrics.setItem(r, 4, _item(m.spec.unit))
            ev = m.evidence or m.note or m.untrusted_why
            self._metrics.setItem(r, 5, _item(ev, tip=ev))
        self._metrics.resizeColumnsToContents()
        self._metrics.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)

    def _fill_skipped(self, card) -> None:
        """未实测 / 不可信的清单 —— **理由必须带出来**。

        ⚠ 理由可能在 `note`（`skip()` 写的）**或** `evidence`（`set(k, None, evidence=...)`
        写的）里 —— 只认一个就会漏（实测：`track.align_ratio` 的理由就在 evidence 里，
        界面于是显示成"（无理由记录）"，而那与"没测"长得一样）。
        """
        lines: List[str] = []
        for d in card.dimensions:
            for m in d.metrics:
                if m.score is None:
                    why = m.note or m.evidence or m.spec.note or "（**没有记录理由**）"
                    lines.append(f"未实测　{m.spec.key}\n　　{why}")
                elif not m.trusted:
                    lines.append(f"不可信　{m.spec.key}\n　　{m.untrusted_why}")
        if not lines:
            lines.append("（没有未实测或不可信的指标）")
        self._skipped.setPlainText("\n\n".join(lines))
