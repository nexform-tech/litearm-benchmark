"""会话与顶层编排 —— 连臂 → 安全前置 → 采集 → 分析 → 打分 → 落盘。

一次运行的完整生命周期只有这一条路径（真机与离线桩走**同一条装配路径**，这是
本家族 core 的既有约定）。任何"顺手加个捷径"都会让离线与真机的行为分叉，别加。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import __version__, instruments as I
from .capture import DualCapture, snapshot_identity
from .metrics.analysis import RawResult, Segment, analyze
from .score import ScoreCard, score
from .suites import tiers as _tiers
from .suites.base import Executor


# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    card: ScoreCard
    raw: RawResult
    segments: List[Segment]
    trace: object
    outdir: Optional[str] = None
    skipped: List[str] = field(default_factory=list)

    def summary(self) -> str:
        t = self.card.total
        head = "（无可计分维度）" if t is None else f"{t:.1f}/100 等级 {self.card.grade}"
        return f"{head}　维度 {len(self.card.scored_dimensions)}/{len(self.card.dimensions)}"


class BenchSession:
    """连臂 + 安全前置 + 限位缓存。**收尾绝不 `disable`**（见 :mod:`.safety`）。"""

    def __init__(self, port: Optional[str] = None, *, dry_run: bool = False,
                 allow_faults: bool = False):
        self.port = port
        self.dry_run = dry_run
        self.allow_faults = allow_faults
        self.arm = None
        self.limits: List[tuple] = []
        self.preflight = None

    def __enter__(self) -> "BenchSession":
        from litearm_core import Arm
        if self.dry_run:
            # ⚠ `port` 必须给个占位：core 的 `connect()` 里端口解析排在传输构造点**之前**，
            #   给 None 会在注入传输口生效之前就报"找不到端口"。
            from litearm_core.testing import FakeTransport
            self.arm = Arm(port=self.port or "fake",
                           transport_factory=lambda p: FakeTransport(p)).connect()
        else:
            self.arm = Arm(port=self.port).connect()

        st = self.arm.get_state(refresh=True).value
        if st is None:
            raise RuntimeError("连上了但读不到状态帧 ⇒ 拒绝继续")
        self.limits = [(p.q_min, p.q_max) for p in self.arm.params.all_joint_params()]
        if not self.limits:
            raise RuntimeError("读不到关节软限位（0x24）⇒ 安全闸无法工作，拒绝继续")

        from .safety import require
        self.preflight = require(self.arm, limits=self.limits,
                                 allow_faults=self.allow_faults)
        # ⚠ **dry-run 也要 enable**：core 的桩传输认 `enable()`（使能后状态帧报 enabled、
        #   300Hz 日志也真能录）⇒ 真机与离线走**同一条装配路径**。在 dry-run 分支里
        #   跳过 enable 会让离线路径与真机路径分叉，那样离线测出来的东西就不代表真机。
        self.arm.enable()
        # 使能后复查：enable 会把 q_ref 设成当前实测位（臂不会跳），但**锁存轴会让整臂
        # 静默降级**，而"回落"这件事在数据上不可观测 ⇒ 宁可在这里硬停。
        st2 = self.arm.get_state(refresh=True).value
        if st2 is not None and getattr(st2, "joint_fault", 0) and not self.allow_faults:
            from .safety import SafetyAbort
            raise SafetyAbort(f"使能后出现 joint_fault=0x{st2.joint_fault:04X} ⇒ 中止")
        return self

    def __exit__(self, *exc) -> None:
        if self.arm is None:
            return
        try:
            if not self.dry_run:
                from .safety import park
                park(self.arm)
        finally:
            self.arm.close()

    # ------------------------------------------------------------------ #
    def run_tier(self, tier: str = _tiers.DEFAULT_TIER, *,
                 interactive: bool = True, outdir: Optional[str] = None,
                 on_log=None) -> RunResult:
        actions = _tiers.actions_of(tier)
        if any(a.is_manual for a in actions) and not interactive:
            raise RuntimeError(f"档位 {tier!r} 含需人工介入的动作，但当前是**非交互**模式 ⇒ 拒绝执行")

        skipped: List[str] = []
        with DualCapture(self.arm) as cap:
            # 记下 flags 位的名字 → 下标，供分析期解码（core 的 `flag_names` 与 bit 一一对应）
            st0 = self.arm.get_state(refresh=False).value
            if st0 is not None:
                cap.trace.meta["flag_names"] = list(getattr(st0, "flag_names", ()) or ())
            cap.trace.meta["tier"] = tier
            # ⚠ 把"这是仿真数据"刻进 trace：报告与**重放**都要靠它拒绝出总分
            cap.trace.meta["simulated"] = bool(self.dry_run)
            ex = Executor(self, cap, interactive=interactive, log=on_log)
            ex.run(actions, deadline=_deadline(_tiers.budget_of(tier)))
            segs = list(ex.segments)
            cap.trace.meta["reconnects"] = ex.reconnects

        # 事后统一补算「全程」类量（它们要等 trace 收全）
        _post_fill(cap.trace, self.limits)

        identity = snapshot_identity(self.arm)
        raw = analyze(cap.trace, segs, identity,
                      tcp_line=_tcp_dev(cap.trace, "cart_line"),
                      tcp_arc=_tcp_dev(cap.trace, "cart_arc"),
                      bench=cap.trace.meta.get("kin_bench"))
        # `analyze` 是从 trace 算的；把 suite 在运行期直接量到的量合并进来
        for k in ("g_match_pct", "watchdog_latency_s", "min_margin_rad",
                  "pos_violation_events", "static_drift_mrad"):
            if k in cap.trace.meta and k.replace("g_match_pct", "torque.g_match_pct") not in raw.values:
                key = {"g_match_pct": "torque.g_match_pct"}.get(k, k)
                if raw.values.get(key) is None and cap.trace.meta[k] is not None:
                    raw.values[key] = float(cap.trace.meta[k])
                    raw.evidence.setdefault(key, f"运行期直接量到（trace.meta['{k}']）")
        card = score(raw, identity=identity, simulated=bool(self.dry_run))

        res = RunResult(card=card, raw=raw, segments=segs, trace=cap.trace,
                        outdir=outdir, skipped=skipped)
        if outdir:
            from .report import save_run
            save_run(outdir, trace=cap.trace, raw=raw, card=card,
                     extra={"tier": tier, "segments": [_seg_dump(s) for s in segs]})
        return res


# --------------------------------------------------------------------------- #
def _deadline(budget: Optional[int]) -> Optional[float]:
    return (time.monotonic() + budget) if budget else None


def _post_fill(trace, limits) -> None:
    """收全 trace 之后才能算的：最小限位裕度、位置越限事件数。"""
    if trace.q.size and limits:
        n = min(trace.q.shape[1], len(limits))
        lo = np.asarray([limits[i][0] for i in range(n)], float)
        hi = np.asarray([limits[i][1] for i in range(n)], float)
        marg = np.minimum(trace.q[:, :n] - lo, hi - trace.q[:, :n])
        trace.meta["min_margin_rad"] = float(np.nanmin(marg))
    # POSITION_VIOLATION 的**上升沿**次数。位名字来自 core 的 flag_names（与 bit 一一对应），
    # 不硬编码位号 —— 布局一变硬编码就静默错。
    names = trace.meta.get("flag_names") or []
    bits = [i for i, nm in enumerate(names) if "POSITION" in str(nm).upper()]
    if bits and trace.flags.size > 1:
        f = trace.flags.astype(np.int64)
        ev = 0
        for b in bits:
            cur = (f >> b) & 1
            ev += int(np.sum((cur[1:] == 1) & (cur[:-1] == 0)))
        trace.meta["pos_violation_events"] = ev


def _tcp_dev(trace, kind: str) -> Optional[Dict[str, object]]:
    """把 TCP 采样与**理论路径**比，给出最大偏离（mm）。

    理论路径用**端点连线**（直线）或**三点定圆的径向偏差**（圆弧）——
    两者都只依赖 TCP 采样本身，不需要运动学模型（这正是"基于 litearm-core"的取舍）。
    """
    if trace.tcp_t.size < 10:
        return None
    # TCP 采样是混在一起的 —— 用时间窗无法区分线段归属，故这里只在**单段**场景下有效。
    # 这是已知限制：多次笛卡尔动作会共用一份采样。报告里会注明。
    p = trace.tcp_p
    if kind == "cart_line":
        a, b = p[0], p[-1]
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 1e-6:
            return None
        d = np.cross(np.tile(ab, (len(p), 1)), p - a) / L
        dev = float(np.linalg.norm(d, axis=1).max() * 1000.0)
        return {"dev_mm": dev,
                "evidence": f"TCP 采样 {len(p)} 点 vs 端点连线（弦长 {L*1000:.1f} mm）；"
                            "⚠ 多点共用一份采样，多次笛卡尔段会互相污染"}
    # 圆弧：三点定圆，取径向偏差
    if len(p) >= 3:
        a, b, c = p[0], p[len(p) // 2], p[-1]
        u, v = b - a, c - a
        nrm = np.cross(u, v)
        nn = float(np.linalg.norm(nrm))
        if nn < 1e-9:
            return None
        return {"dev_mm": None,
                "evidence": "圆弧三点近共线 ⇒ 定不出圆（这一段的径向偏差不给分）"}
    return None


def _seg_dump(s: Segment) -> Dict[str, object]:
    return {"label": s.label, "kind": s.kind, "t0": s.t0, "t1": s.t1,
            "target": s.target, "chunk": s.chunk, "extra": s.extra}


# --------------------------------------------------------------------------- #
def run(tier: str = _tiers.DEFAULT_TIER, *, port: Optional[str] = None,
        outdir: Optional[str] = None, dry_run: bool = False,
        interactive: bool = True, on_log=None) -> RunResult:
    """一次完整基准运行。``outdir`` 给了就落盘（三层：npz/json/md）。"""
    if outdir is None:
        stamp = time.strftime("%Y%m%d_%H%M")
        outdir = os.path.join("docs", f"bench_{stamp}_{tier}")
    with BenchSession(port, dry_run=dry_run) as s:
        return s.run_tier(tier, interactive=interactive, outdir=outdir, on_log=on_log)
