"""打分 —— 原始值 → 指标分 → 维度分 → 总分/等级。

三条规则，每条都对应本项目踩过的一个坑：

1. **未实测 ≠ 0 分**。测不出来（无笛卡尔、无真机、未跑负载档）就不参与聚合，
   报告里单列「未实测」，而不是用 0 分把它拖下去。总分按**实际测到的维度重新归一化**，
   并显式列出用了哪些权重。
2. **不可信 ≠ 高分**。跨流类指标（`track.*`）若自证量不达标，**整组剔除**并标注原因 ——
   本项目历史上吃过「对齐失效却照常出数」的亏（那批 30~133 mrad 的记录就是）。
3. **等级只由总分决定，且分档写死**，不给"调权重凑等级"留空间。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .metrics.analysis import RawResult
from .metrics.registry import (
    ALL_METRICS, BY_KEY, DIMENSION_ORDER, DIMENSION_WEIGHTS, MetricSpec,
)

#: 「自证量不达标就不许出分」的指标组：`<前缀>` → 它的自证指标 key
_TRUST_GATE: Dict[str, str] = {"track.": "track.align_ratio"}

GRADES: Tuple[Tuple[float, str], ...] = (
    (90.0, "S"), (80.0, "A"), (70.0, "B"), (60.0, "C"), (40.0, "D"), (0.0, "F"),
)


@dataclass
class MetricScore:
    spec: MetricSpec
    raw: Optional[float]
    score: Optional[float]
    evidence: str = ""
    note: str = ""
    trusted: bool = True
    untrusted_why: str = ""

    @property
    def measured(self) -> bool:
        return self.score is not None

    @property
    def counted(self) -> bool:
        return self.score is not None and self.trusted


@dataclass
class DimensionScore:
    name: str
    weight: float
    metrics: List[MetricScore] = field(default_factory=list)

    @property
    def counted(self) -> List[MetricScore]:
        return [m for m in self.metrics if m.counted]

    @property
    def score(self) -> Optional[float]:
        c = self.counted
        if not c:
            return None
        wsum = sum(m.spec.weight for m in c)
        if wsum <= 0:
            return None
        return sum((m.score or 0.0) * m.spec.weight for m in c) / wsum

    @property
    def measured_ratio(self) -> str:
        return f"{len(self.counted)}/{len(self.metrics)}"


@dataclass
class ScoreCard:
    dimensions: List[DimensionScore] = field(default_factory=list)
    identity: Dict[str, object] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)
    #: **仿真/桩数据跑出来的** ⇒ 一律不出总分。
    #: 理由：桩不建模运动 ⇒ `q` 恒定 ⇒ 抖动 0 ⇒ 满分；`τ=0` ⇒ 力矩占比 0 ⇒ 满分。
    #: 这种"什么都没测却得 78 分"是最危险的假绿，必须在**聚合层**堵死，而不是靠各指标自觉。
    simulated: bool = False
    #: 本次运行的**激励**（臂到底动没动）。运动类指标的出分前提。
    excitation: Dict[str, float] = field(default_factory=dict)

    @property
    def scored_dimensions(self) -> List[DimensionScore]:
        if self.simulated:
            return []
        return [d for d in self.dimensions if d.score is not None]

    @property
    def total(self) -> Optional[float]:
        ds = self.scored_dimensions
        if not ds:
            return None
        w = sum(d.weight for d in ds)
        return sum((d.score or 0.0) * d.weight for d in ds) / w if w > 0 else None

    @property
    def grade(self) -> str:
        t = self.total
        if t is None:
            return "—"
        for lo, g in GRADES:
            if t >= lo:
                return g
        return "F"

    @property
    def weights_used(self) -> Dict[str, float]:
        return {d.name: d.weight for d in self.scored_dimensions}

    @property
    def missing_dimensions(self) -> List[str]:
        return [d.name for d in self.dimensions if d.score is None]

    def metric(self, key: str) -> Optional[MetricScore]:
        for d in self.dimensions:
            for m in d.metrics:
                if m.spec.key == key:
                    return m
        return None


# --------------------------------------------------------------------------- #
def score(raw: RawResult, *, identity: Optional[Dict[str, object]] = None,
          simulated: bool = False) -> ScoreCard:
    """把 :class:`RawResult` 变成成绩单。纯函数 —— 同样的 raw 必然得到同样的分。

    ``simulated=True`` 时**不产生任何总分**：仿真数据能让"什么都没测"看起来像满分
    （桩的 `q` 恒定 ⇒ 抖动 0 ⇒ 100 分），这是最危险的假绿，故在聚合层直接堵死。
    """
    card = ScoreCard(identity=dict(identity or {}), simulated=simulated)
    if simulated:
        card.simulated = True
        raw.notes["_simulated"] = ("⚠ 本次为 **dry-run / 仿真**数据 ⇒ 不出总分。"
                                   "桩传输**不建模运动**，运动类指标的退化读数（抖动 0、τ=0）"
                                   "会伪装成满分。要分数请接真机。")
    card.notes.update(raw.notes)

    # 自证闸：先决定哪些组"不可信"
    untrusted: Dict[str, str] = {}
    for prefix, gate_key in _TRUST_GATE.items():
        gv = raw.values.get(gate_key)
        spec = BY_KEY.get(gate_key)
        gs = spec.score(gv) if spec else None
        if gv is None:
            untrusted[prefix] = f"自证量 {gate_key} 未测出"
        elif gs is not None and (gv < 2.0):
            untrusted[prefix] = (f"自证量 {gate_key}={gv:.2f}× < 2.0× ⇒ 该组数据不可信，"
                                 f"整组剔除（不是给低分）")

    by_dim: Dict[str, List[MetricScore]] = {d: [] for d in DIMENSION_ORDER}
    for spec in ALL_METRICS:
        rv = raw.values.get(spec.key)
        ms = MetricScore(spec=spec, raw=rv, score=spec.score(rv),
                         evidence=raw.evidence.get(spec.key, ""),
                         note=raw.notes.get(spec.key, ""))
        for prefix, why in untrusted.items():
            if spec.key.startswith(prefix):
                ms.trusted = False
                ms.untrusted_why = why
        if simulated:
            # 仿真下**逐条**判不可信 —— 只把总分置空是不够的：控制台与报告里
            # 维度分仍会显示成好看的数字，读者照样会被误导。
            ms.trusted = False
            ms.untrusted_why = "仿真数据（桩不建模运动，退化读数会伪装成满分）⇒ 不出分"
        by_dim.setdefault(spec.dimension, []).append(ms)

    for name in DIMENSION_ORDER:
        ms = by_dim.get(name) or []
        if not ms:
            continue
        card.dimensions.append(DimensionScore(name=name, weight=DIMENSION_WEIGHTS.get(name, 1.0),
                                              metrics=ms))
    for d in card.dimensions:
        if d.score is None:
            why = "全部未实测"
            unt = [m for m in d.metrics if not m.trusted]
            if unt and len(unt) == len(d.metrics):
                why = unt[0].untrusted_why
            card.notes[f"dim:{d.name}"] = f"不参与总分（{why}）"
    return card


def grade_of(total: Optional[float]) -> str:
    if total is None:
        return "—"
    for lo, g in GRADES:
        if total >= lo:
            return g
    return "F"
