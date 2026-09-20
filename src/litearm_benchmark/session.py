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
                 on_log=None, on_prompt=None) -> RunResult:
        """跑一个**档位** —— 动作表与挂钟预算都取自 :mod:`.suites.tiers`。"""
        return self.run_actions(_tiers.actions_of(tier), label=tier,
                                budget=_tiers.budget_of(tier),
                                interactive=interactive, outdir=outdir,
                                on_log=on_log, on_prompt=on_prompt)

    def run_actions(self, actions, *, label: str = "custom",
                    budget: Optional[int] = None,
                    interactive: bool = True, outdir: Optional[str] = None,
                    on_log=None, on_prompt=None) -> RunResult:
        """跑**任意一组动作**。

        ⚠ **档位与"单项"必须走这同一条装配路径** —— 本家族最忌讳"GUI/单项自己
        另起一套采集与分析"。分叉之后两边测出来的东西就不再可比，而可比性正是
        这个基准的全部价值。`run_tier` 只是它带上档位动作表的一个薄包装。

        `budget` 是**截断死线**（秒），超了就停跑剩余动作；`None` = 不限。
        """
        actions = list(actions)
        if any(a.is_manual for a in actions) and not interactive:
            raise RuntimeError(f"{label!r} 含需人工介入的动作，但当前是**非交互**模式 ⇒ 拒绝执行")

        skipped: List[str] = []
        with DualCapture(self.arm) as cap:
            # 记下 flags 位的名字 → 下标，供分析期解码（core 的 `flag_names` 与 bit 一一对应）
            st0 = self.arm.get_state(refresh=False).value
            if st0 is not None:
                cap.trace.meta["flag_names"] = list(getattr(st0, "flag_names", ()) or ())
            cap.trace.meta["tier"] = label
            # ⚠ 把"这是仿真数据"刻进 trace：报告与**重放**都要靠它拒绝出总分
            cap.trace.meta["simulated"] = bool(self.dry_run)
            ex = Executor(self, cap, interactive=interactive, log=on_log,
                          prompt=on_prompt)
            ex.run(actions, deadline=_deadline(budget))
            segs = list(ex.segments)
            cap.trace.meta["reconnects"] = ex.reconnects

        # 事后统一补算「全程」类量（它们要等 trace 收全）
        _post_fill(cap.trace, self.limits)
        _link_counters(cap.trace, self.arm)

        identity = snapshot_identity(self.arm)
        raw = analyze(cap.trace, segs, identity,
                      bench=cap.trace.meta.get("kin_bench"))
        # ⚠ 这里**曾经**有一段"把 trace.meta 合并进 raw.values"的循环，已删：
        #   它的守卫是 `... and <目标键> not in raw.values`，而 `skip()` 会把键
        #   **以 None 写进** `raw.values` ⇒ `not in` **恒假** ⇒ 那一支永远不执行。
        #   它想服务的 5 个键里，4 个（`min_margin_rad` / `pos_violation_events` /
        #   `static_drift_mrad` / `watchdog_latency_s`）本来就是各分析函数**自己读
        #   `trace.meta`** 的（那段循环对它们是死代码）；唯一真需要它的
        #   `g_match_pct` 于是**从来没被合并过**（真机每档都判未实测）。
        #   ⇒ 统一改成分析侧直接读 meta，一处真相，不再有第二遍合并。
        card = score(raw, identity=identity, simulated=bool(self.dry_run))

        res = RunResult(card=card, raw=raw, segments=segs, trace=cap.trace,
                        outdir=outdir, skipped=skipped)
        if outdir:
            from .report import save_run
            save_run(outdir, trace=cap.trace, raw=raw, card=card,
                     extra={"tier": label, "segments": [_seg_dump(s) for s in segs]})
        return res


# --------------------------------------------------------------------------- #
def _deadline(budget: Optional[int]) -> Optional[float]:
    return (time.monotonic() + budget) if budget else None


def _post_fill(trace, limits) -> None:
    """收全 trace 之后才能算的：最小限位裕度、位置越限事件数。"""
    if trace.q.size and limits:
        n = min(trace.q.shape[1], len(limits))
        lo = np.asarray([limits[i][0] for i in range(n)], float) - I.POS_LATCH_BAND.value
        hi = np.asarray([limits[i][1] for i in range(n)], float) + I.POS_LATCH_BAND.value
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


def _link_counters(trace, arm) -> None:
    """core 侧的丢帧计数（**会话结束时的累计快照**）。

    ⚠ 这两个是 **core 的**计数器（挂在 `_Ack` 上），**不是固件的** —— 所以它们
    跟 `kin_bench` 没有任何关系。旧实现只在 `_kin_bench` 里找它们，
    于是 `link.foreign_frames` **永远取不到**（那一项恒判未实测）。

    * `foreign_frames` —— "认识、但不是本命令要的那条"被某个读循环丢掉的帧数
      （八处读循环各计各的）；
    * `unexpected_frames` —— 未识别 id 的帧数。

    两者期望恒 0。⚠ 本项目记录过一次真实事故：真机把多帧合成一帧，于是
    "计数器恒 0" 被读成了 "没出错" —— 所以这里**只在真拿到属性时才写**，
    拿不到就让它保持未实测，而不是填 0。
    """
    a = getattr(arm, "_a", None)
    if a is None:
        return
    for key in ("foreign_frames", "unexpected_frames"):
        v = getattr(a, key, None)
        if v is not None:
            trace.meta[key] = int(v)


def _seg_dump(s: Segment) -> Dict[str, object]:
    return {"label": s.label, "kind": s.kind, "t0": s.t0, "t1": s.t1,
            "target": s.target, "chunk": s.chunk, "extra": s.extra,
            "travel_rad": s.travel_rad}


# --------------------------------------------------------------------------- #
def run(tier: str = _tiers.DEFAULT_TIER, *, port: Optional[str] = None,
        outdir: Optional[str] = None, dry_run: bool = False,
        interactive: bool = True, on_log=None, on_prompt=None) -> RunResult:
    """一次完整基准运行。``outdir`` 给了就落盘（三层：npz/json/md）。"""
    if outdir is None:
        stamp = time.strftime("%Y%m%d_%H%M")
        outdir = os.path.join("docs", f"bench_{stamp}_{tier}")
    with BenchSession(port, dry_run=dry_run) as s:
        return s.run_tier(tier, interactive=interactive, outdir=outdir, on_log=on_log,
                          on_prompt=on_prompt)


def run_actions(actions, *, label: str = "custom", budget: Optional[int] = None,
                port: Optional[str] = None, outdir: Optional[str] = None,
                dry_run: bool = False, interactive: bool = True,
                on_log=None, on_prompt=None) -> RunResult:
    """跑**单个测试项**（任意一组动作）—— 与 :func:`run` **同一条装配路径**。

    ⚠ 这条路径存在的意义是**别让 GUI/单项另起一套**：分叉之后两边测出来的东西
    就不再可比，而可比性正是这个基准的全部价值。安全闸（`preflight` / `_gate`）
    在 `BenchSession` 与 `Executor` 里，**一视同仁**。
    """
    if outdir is None:
        stamp = time.strftime("%Y%m%d_%H%M")
        outdir = os.path.join("docs", f"bench_{stamp}_{label}")
    with BenchSession(port, dry_run=dry_run) as s:
        return s.run_actions(actions, label=label, budget=budget,
                             interactive=interactive, outdir=outdir,
                             on_log=on_log, on_prompt=on_prompt)
