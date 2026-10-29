"""指标计算 —— 从 :class:`~litearm_benchmark.capture.Trace` 算出**原始值**。

本模块**只产生原始值**（`Dict[key, float]`），从不算分；分数与阈值在 :mod:`.registry`。
这条分界是刻意的：它让「改权重/改阈值」不需要重跑硬件（见 :mod:`..replay`）。

三条贯穿全模块的口径纪律：

* **未实测 ≠ 得零分**。算不出来就放进 ``None``，报告里标「未实测」。
* **先扣地板再判抖动**。力矩/速度的量化地板在 :mod:`..instruments` 里，不扣就是把
  台阶当抖动。
* **跨流的数必须带自证量**。`track.*` 全部依赖 `align`，所以 **`track.align_ratio`
  不达标时把 `track` 那几条一并标为不可信**，而不是照常出分。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import instruments as I
from ..align import align, moving_axes
from . import signals as S


@dataclass
class Segment:
    """一个被观测的动作段（由 suite 在跑的时候登记）。

    `t0/t1` 是 **100Hz 时间轴**上的起止；`chunk` 指向对应的 300Hz 日志窗口下标。
    """

    label: str
    kind: str                                   # movej / movej_sync / move_l / move_c / move_path / rest
    t0: float
    t1: float
    target: Optional[List[float]] = None
    chunk: Optional[int] = None
    extra: Dict[str, object] = field(default_factory=dict)
    #: 本段**实测**最大单轴行程（rad）—— 由执行器在动臂时量（`Executor._joint` 填）。
    #: ``None`` = 没记录（旧数据、或本来就是静止段）⇒ 分析侧**跳过该段**，不猜。
    #:
    #: ⚠ 为什么必须逐段量：`MIN_EXCITATION_RAD` 原来只有**全程**一版，于是
    #: "重复 3 次"这种动作里，若臂**根本没动**（目标恰好等于上一个动作的落点），
    #: 三次落点的差就只是**传感器噪声** —— 2026-09-20 真机首跑实测：`gcheck` 的
    #: 最后一个位形就是 `bench_j2lo`，随后 `小位移@0.3` 的三次"重复"段长全是 0.02 s，
    #: 而 `static.repeat_ptp_mrad` 拿着那 0.76 mrad **判了 91 分**。
    travel_rad: Optional[float] = None


#: 一旦"臂基本没动"，这些指标全部判未实测（它们全是运动派生量）。
_MOTION_KEYS = (
    "track.rms_mrad", "track.peak_mrad", "track.align_ratio", "track.lag_ms",
    "track.follow_p95_mrad",
    "dyn.overshoot_mrad", "dyn.rebound_mrad", "dyn.settle_s", "dyn.rise_ms",
    "smooth.vel_ripple_pct", "smooth.jerk_spike_ratio",
    "smooth.tau_sigma_nm", "smooth.tau_vs_quant",
    "torque.peak_ratio", "torque.saturation_frac", "torque.g_match_pct",
    "cart.line_dev_mm", "cart.arc_dev_mm", "cart.plan_us",
    "sync.stop_spread_ms",
)

#: 激励闸：全程最大单轴行程低于此值 ⇒ 一切**运动类**指标判为未实测。
#: 20 mrad ≈ 位置编码器 LSB 的 200 倍，也远小于任何一个有意义动作的行程。
MIN_EXCITATION_RAD = 0.02


def segment_excited(s: Segment) -> Optional[bool]:
    """单段版激励闸：这一段**实测**动了吗？

    * ``True``  —— 行程 ≥ :data:`MIN_EXCITATION_RAD`；
    * ``False`` —— 动了但不够（目标等于起点、或只抖了一下）；
    * ``None``  —— **没记录行程**（旧数据）⇒ 调用方**必须跳过**，不许当成 ``True``。

    判据与全程版**共用同一个阈值** —— "动了"的标准不该因为量程不同而变。
    """
    return None if s.travel_rad is None else (float(s.travel_rad) >= MIN_EXCITATION_RAD)


@dataclass
class RawResult:
    values: Dict[str, Optional[float]] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)
    evidence: Dict[str, str] = field(default_factory=dict)
    #: 全程最大单轴行程（rad）—— 由 :func:`analyze` 填，供报告与闸门用。
    excitation_rad: float = 0.0

    @property
    def excited(self) -> bool:
        return self.excitation_rad >= MIN_EXCITATION_RAD

    def set(self, key: str, value: Optional[float], evidence: str = "",
            note: str = "") -> None:
        self.values[key] = None if value is None else float(value)
        if evidence:
            self.evidence[key] = evidence
        if note:
            self.notes[key] = note

    def skip(self, key: str, why: str) -> None:
        self.values[key] = None
        self.notes[key] = why


# --------------------------------------------------------------------------- #
def analyze(trace, segments: Sequence[Segment], identity: Dict[str, object],
            *, bench: Optional[dict] = None) -> RawResult:
    """总入口。``bench`` 是固件自检结果（来自 `diag.kin_bench()`）的字典。"""
    r = RawResult()
    segs = list(segments)

    # ---- 激励存在性闸（**先立后破**）----
    # 运动类指标全都建立在"臂真的动过"之上。臂没动时它们会给出**退化但有限**的读数
    # （抖动 0、τ 恒 0 ⇒ 满分），这是最危险的一种假绿。所以在入口处量一次行程，
    # 不够就直接把所有运动类指标标成"未实测"。
    travel = 0.0
    if trace.q.size and trace.q.shape[0] >= 2:
        travel = float(np.ptp(trace.q, axis=0).max())
    r.evidence["_excitation"] = f"全程最大单轴行程 {travel * 1000:.2f} mrad"
    r.excitation_rad = travel

    _static(r, trace, segs, identity)
    if r.excited:
        _tracking(r, trace, segs)
        _dynamic(r, trace, segs)
        _torque(r, trace, segs, identity)
        _cartesian(r, trace, segs)
        _sync(r, trace, segs)
    else:
        why = (f"**激励不足**：全程最大单轴行程只有 {travel * 1000:.2f} mrad "
               f"< {MIN_EXCITATION_RAD * 1000:.0f} mrad ⇒ 臂基本没动，运动类指标一律判未实测")
        for k in _MOTION_KEYS:
            r.skip(k, why)
    # 平滑度里**静止抖动**那两条不需要运动（臂不动时的抖动本身就是有效测量），
    # 所以传 excited 进去由它自己决定跳哪些子项。
    _smoothness(r, trace, segs, identity, excited=r.excited)
    _thermal(r, trace)
    _endurance(r, trace)
    _safety(r, trace, identity)
    _compute(r, bench)
    _link(r, trace, identity)
    return r


# --------------------------------------------------------------------------- #
def _rest_hold_windows(t: np.ndarray, q: np.ndarray, segs: Sequence[Segment]):
    """把每个 `rest`/`hold` 段切成 (标号, 时间, 位置) 三元组。"""
    out = []
    for s in segs:
        if s.kind not in ("rest", "hold"):
            continue
        m = (t >= s.t1 - 0.6) & (t <= s.t1 - 0.05)      # 取段末去掉过渡的窗口
        if int(m.sum()) >= 5:
            out.append((s, t[m], q[m]))
    return out


# ============================ A. 静态精度与重复性 ============================ #
def _static(r: RawResult, trace, segs, identity) -> None:
    holds = _rest_hold_windows(trace.t_status, trace.q, segs)
    if not holds:
        for k in ("static.err_max_mrad", "static.repeat_ptp_mrad", "static.floor_ratio"):
            r.skip(k, "没有静止段 ⇒ 静差类指标无法测")
        return

    n = trace.n_joints
    lim = _limits_from_identity(identity, n)

    # 目标一律**先 clamp**：目标越软限时固件会把锚点钳回限内，拿未钳目标比会**假红**
    # （本项目真机实测过：J4 目标 +0.0193 越上限 +0.0175 ⇒ 对未钳目标差 −2.67 mrad「FAIL」，
    #   对钳后目标只差 −0.76 mrad「PASS」）。
    worst_err, worst_axis = 0.0, 0
    for s, _t, qw in holds:
        if not s.target:
            continue
        tgt = np.asarray(s.target, float)
        if lim is not None:
            lo, hi = lim
            tgt = np.clip(tgt, lo, hi)
        dev = np.abs(qw.mean(axis=0) - tgt) * 1000.0        # mrad
        i = int(np.argmax(dev))
        if dev[i] > worst_err:
            worst_err, worst_axis = float(dev[i]), i + 1
    r.set("static.err_max_mrad", worst_err,
          evidence="静止段落点均值 vs clamp 后的指令目标，逐段取最大",
          note=f"最差 J{worst_axis}；共 {len(holds)} 个静止段")

    # 重复性：**同一目标**的多段之间取极差。
    #
    # ⚠ **只有"真动过"的落点才算**（逐段激励闸）。不动的话，同一目标的多次落点之间
    #   的差只是**传感器噪声** —— 那正是 2026-09-20 真机首跑里那个 0.76 mrad「91 分」
    #   的来历：`gcheck` 已把臂停在 `bench_j2lo`，随后三次"重复"一次都没动。
    by_tgt: Dict[tuple, List[np.ndarray]] = {}
    n_used = 0
    for s, _t, qw in holds:
        if not s.target:
            continue
        if segment_excited(s) is not True:      # False(没动) 与 None(没记录) 一律不算
            continue
        key = tuple(round(v, 4) for v in np.asarray(s.target, float))
        by_tgt.setdefault(key, []).append(qw.mean(axis=0))
        n_used += 1

    reps = {k: v for k, v in by_tgt.items() if len(v) >= 2}
    if reps:
        ptp = max(float(np.ptp(np.stack(arrs), axis=0).max() * 1000.0)
                  for arrs in reps.values())
        r.set("static.repeat_ptp_mrad", ptp,
              evidence=f"同一目标多轮之间的极差；{n_used} 个**真动过**的落点参与，"
                       f"{len(reps)} 个目标有 ≥2 次")
    else:
        # ⛔ **不许退化成"段内样本极差"** —— 那是**噪声**不是重复定位精度。
        #   拿它打分就是拿一个别的量冒充（本项目"未实测 ≠ 编个数"那条规矩）。
        r.skip("static.repeat_ptp_mrad",
               f"没有**真动过**的重复落点（共 {len(holds)} 个静止段、"
               f"{n_used} 个真动过）⇒ 重复定位精度**测不了**，"
               f"而不是「测出来很好」（静止臂的落点差只是传感器噪声）")

    # 物理地板归一化
    fc0 = identity.get("fc0"); fc1 = identity.get("fc1")
    kp = _kp_hold_from_identity(identity)
    if fc0 and fc1 and kp:
        floors = [I.static_error_floor_rad(fc0[i], fc1[i], kp[i]) * 1000.0
                  for i in range(min(len(fc0), len(kp)))]
        j = worst_axis - 1
        if 0 <= j < len(floors) and np.isfinite(floors[j]) and floors[j] > 0:
            r.set("static.floor_ratio", worst_err / floors[j],
                  evidence=f"静差 {worst_err:.2f} / 地板 {floors[j]:.2f} mrad（J{worst_axis}）")
        else:
            # ⚠ `kp_hold == 0` ⇒ 地板无界。**必须跳过而不是当 0** —— 当 0 会让这个比值
            #   成为 0 分，把"没数据"伪装成"性能极差"。
            r.skip("static.floor_ratio",
                   "地板无界或为 0（kp_hold ≤ 0、或读不到 fc0/fc1）⇒ 该比值不可用")
        # 顺带把地板本身作为证据留下（报告里展示"为什么这个静差是好的")
        r.evidence["static._floor_mrad"] = ", ".join(f"{v:.2f}" for v in floors)
    else:
        r.skip("static.floor_ratio", "读不到 fc0/fc1 或 mit_kp/hold_kp_gain ⇒ 算不出物理地板")


# ============================== B. 位置跟踪 ================================= #
def _tracking(r: RawResult, trace, segs) -> None:
    keys = ("track.rms_mrad", "track.peak_mrad", "track.align_ratio",
            "track.lag_ms", "track.follow_p95_mrad")
    results = []
    n_noexc = 0
    for s in segs:
        if s.chunk is None or s.kind not in ("movej", "movej_sync", "move_l", "move_c"):
            continue
        # ⚠ **逐段激励闸**：目标等于上一个动作落点的"空动作"段长只有 0.02 s，
        #   里面的 31 个帧全是静止样本 —— 它过得了"≥30 帧"这条代理判据，
        #   却**不含任何跟踪信息**，对齐只会给出 NaN。按行程判才是问对了问题。
        if segment_excited(s) is not True:
            n_noexc += 1
            continue
        m = (trace.t_status >= s.t0) & (trace.t_status <= s.t1 + 0.3)
        if int(m.sum()) < 30:
            continue
        tl, qr, _dq, _tau = trace.log_arrays(s.chunk)
        if len(tl) < 30:
            continue
        ax = moving_axes(tl, qr)
        res = align(trace.t_status[m], trace.q[m], tl, qr, axes=ax)
        results.append((s, res, m, tl, qr))
    if not results:
        why = (f"没有可对齐的**运动**段（{n_noexc} 段行程不足 "
               f"{MIN_EXCITATION_RAD * 1000:.0f} mrad 被逐段激励闸剔除）"
               if n_noexc else
               "没有可对齐的运动段（需要 300Hz 日志窗口 + ≥30 个状态帧）")
        for k in keys:
            r.skip(k, why)
        return

    good = [x for x in results if x[1].ok]
    _exc = (f"；另有 {n_noexc} 段行程不足 {MIN_EXCITATION_RAD * 1000:.0f} mrad，"
            f"逐段激励闸未放行") if n_noexc else ""
    r.set("track.rms_mrad",
          (max(x[1].rms for x in good) * 1000.0) if good else None,
          evidence=f"逐段互相关对齐后取**最差**一段；{len(good)}/{len(results)} 段自证达标"
                   + _exc)
    r.set("track.peak_mrad",
          (max(x[1].peak for x in good) * 1000.0) if good else None,
          evidence="同上的残差峰值")
    # ⚠ **`min` 前必须先滤掉 NaN**：两流无关时 `control_rms` 为 NaN ⇒ `ratio` 为 NaN，
    #   而 Python 的 `min` 遇 NaN 的结果**顺序相关**（先遇到谁就留谁）⇒ NaN 会污染整个
    #   `min`，把一个"其实算得出来"的自证量静默变成「未实测」（2026-09-20 真机首跑实测：
    #   3 段有 3.6~6.3× 的有效自证量，却被 5 段的 NaN 淹掉，指标直接不出分）。
    ratios = [x[1].ratio for x in results if np.isfinite(x[1].ratio)]
    r.set("track.align_ratio", min(ratios) if ratios else None,
          evidence=f"**自证量**取最差（1.0≈噪声，真结构 3~6×）；"
                   f"{len(ratios)}/{len(results)} 段算得出自证量"
                   + ("" if ratios else " ⇒ 全部无自证量，不出分"))
    r.set("track.lag_ms",
          float(np.median([x[1].delta for x in good]) * 1000.0) if good else None,
          evidence="各段 δ 的中位数（负=实测领先参考）")

    # 跟随误差 p95 —— 只在自证达标时给
    if good:
        vals = []
        for s, res, m, tl, qr in good:
            from ..align import joined
            _tt, qm, ref = joined(trace.t_status[m], trace.q[m], tl, qr, res)
            vals.append(np.abs(qm - ref).max(axis=1))
        r.set("track.follow_p95_mrad", S.p95(np.concatenate(vals)) * 1000.0,
              evidence="对齐后逐点 |q−q_ref| 的 p95")
    else:
        r.skip("track.follow_p95_mrad",
               f"全部 {len(results)} 段的自证量都不达标 ⇒ 跨流残差不可信，不给分")

    if not good:
        for k in ("track.rms_mrad", "track.peak_mrad", "track.lag_ms"):
            r.skip(k, "没有一段的自证量达标 ⇒ 拒绝出分（而不是给个假的）")

    bad = [x[1].why_bad() for x in results if not x[1].ok]
    if bad:
        r.notes["track._bad_segments"] = "；".join(sorted(set(bad))[:3])


# ============================== C. 动态响应 ================================= #
def _dynamic(r: RawResult, trace, segs) -> None:
    ovs, rbs, sts = [], [], []
    rises = []                                   # (指令速度, 上升时间 ms) 对
    for s in segs:
        if s.kind not in ("movej", "movej_sync") or not s.target:
            continue
        m = (trace.t_status >= s.t0) & (trace.t_status <= s.t1 + 0.3)
        if int(m.sum()) < 8:
            continue
        t, q = trace.t_status[m], trace.q[m]
        tgt = np.asarray(s.target, float)
        q0 = q[0]
        delta = tgt - q0
        ctr = np.where(np.abs(delta) > 0.05)[0]
        if not len(ctr):
            continue
        j = int(ctr[np.argmax(np.abs(delta[ctr]))])
        ov, rb = S.overshoot_rebound(q[:, j], float(tgt[j]), float(np.sign(delta[j])))
        ovs.append(ov * 1000.0); rbs.append(rb * 1000.0)
        st = S.settle_time(t, q[:, j], float(tgt[j]), band=0.01, t0=float(t[0]))
        if np.isfinite(st):
            sts.append(st)
        rt = S.rise_time(t, q[:, j], float(q0[j]), float(tgt[j]), float(t[0]))
        if np.isfinite(rt):
            rises.append((float((s.extra or {}).get("speed") or 0.0), rt * 1000.0))

    def put(key, arr, ev):
        if arr:
            r.set(key, float(np.median(arr)), ev)
        else:
            r.skip(key, "没有可用的关节运动段")

    put("dyn.overshoot_mrad", ovs, "各段越峰的中位数（逐段取主运动轴，对**指令目标**取）")
    put("dyn.rebound_mrad", rbs, "各段回摆的中位数（峰值后往回收的幅度，恒 ≥0）")
    put("dyn.settle_s", sts, "各段整定时间中位数（进入 |q−目标| ≤ 10 mrad 并保持）")

    # ⚠⚠ `dyn.rise_ms` **必须同档比** —— 它的出处自己就写着「换速度档会成比例变化
    #    ⇒ 必须同档比」。而旧实现**跨所有速度段取一个中位数** ⇒ 量到的主要是
    #    「命令速度有多慢」，而不是这台臂的响应快慢（真机：同一批数据里 0.3 档与
    #    1.0 档的上升时间差好几倍，混在一起中位数只反映段数最多的那一档）。
    #    ⇒ 按段的**指令速度**分组，只报**最快档**的中位数（跨运行可比），
    #    并把各档分布一并写进证据（换档时能直接看出比例关系）。
    if rises:
        by_sp: Dict[float, List[float]] = {}
        for sp, v in rises:
            by_sp.setdefault(sp, []).append(v)
        top = max(by_sp)
        spread = "、".join(f"speed={sp:g}: {np.median(v):.0f} ms（{len(v)} 段）"
                          for sp, v in sorted(by_sp.items()))
        r.set("dyn.rise_ms", float(np.median(by_sp[top])),
              evidence=f"**最快档**（speed={top:g}）的 10%→90% 行程时间中位数。"
                       f"⚠ **只同档比** —— 它随指令速度成比例变化。各档实测：{spread}")
        r.notes["dyn.rise_ms"] = f"共 {len(by_sp)} 个速度档参与；只报最快档"
    else:
        r.skip("dyn.rise_ms", "没有可用的关节运动段")


# ============================ D. 平滑度与抖动 =============================== #
def _smoothness(r: RawResult, trace, segs, identity, *, excited: bool = True) -> None:
    # ---- 静止抖动（对位置做直线拟合残差）----
    holds = _rest_hold_windows(trace.t_status, trace.q, segs)
    if holds:
        sig = []
        for _s, t, q in holds:
            sg = [S.fit_line_residual_std(t, q[:, j]) for j in range(q.shape[1])]
            sg = [v for v in sg if np.isfinite(v)]
            if sg:
                sig.append(max(sg))                 # 取该段最差的轴（腕部通常最大）
        if sig:
            r.set("smooth.rest_sigma_mrad", float(np.median(sig)) * 1000.0,
                  evidence="逐静止段取最差轴的**直线拟合残差 std**，再取中位数")
            floor = I.Q_LSB_RAD.value / 12 ** 0.5
            r.set("smooth.rest_vs_quant", float(np.median(sig)) / floor,
                  evidence=f"归一化到纯量化锯齿地板 {floor*1000:.4f} mrad（LSB/√12）")
    if "smooth.rest_sigma_mrad" not in r.values:
        r.skip("smooth.rest_sigma_mrad", "没有静止段 ⇒ 抖动无法测")
        r.skip("smooth.rest_vs_quant", "同上")

    # ---- 巡航段的速度纹波 / 力矩纹波 / jerk 尖峰 ----
    # ⚠ 这四条是**运动派生**量：臂没动时它们会给出退化读数（std=0 ⇒ 满分），故先守闸。
    ripples, tau_sig, jerk_spikes = [], [], []
    if not excited:
        for k in ("smooth.vel_ripple_pct", "smooth.jerk_spike_ratio",
                  "smooth.tau_sigma_nm", "smooth.tau_vs_quant"):
            r.skip(k, "激励不足（臂基本没动）⇒ 巡航段不存在")
        return
    n = trace.n_joints
    for s in segs:
        if s.chunk is None or s.kind not in ("movej", "movej_sync"):
            continue
        tl, qr, dqr, taul = trace.log_arrays(s.chunk)
        if len(tl) < 60:
            continue
        ax = moving_axes(tl, qr)
        if not ax:
            continue
        # ⚠ 速度取**位置差分**而不是日志的 dq 列：出厂 ff_mask 含 FF_VELREF 时
        #   dq 列被固件清零（见 instruments.LOG_DQ_IS_COMMANDED），恒 0 不能用。
        for j in ax:
            v = np.gradient(qr[:, j], I.CTRL_DT.value)
            seg_i = S.cruise_segment(v)
            if seg_i is None or seg_i[1] - seg_i[0] < 10:
                continue
            rp = S.ripple_pct(v, *seg_i)
            if np.isfinite(rp):
                ripples.append(rp)
            ts = float(np.std(taul[seg_i[0]:seg_i[1] + 1, j], ddof=1))
            lsb = I.TAU_LSB_NM.get(j + 1)
            tau_sig.append(S.excess_over_floor(ts, S.quant_floor_std(lsb.value)) if lsb else ts)
            # ⚠⚠⚠ **分母必须是「巡航速度」，不能是「典型的一拍变化量 `median|Δv|`」。**
            #
            # 旧口径（`max|Δv| / median|Δv|`）的量级由**参考平滑到什么程度**决定：
            # 参考越平滑，`median|Δv|` 越接近 **f32 舍入噪声** —— 6 小时浸泡实测它低到
            # **3.143e-07**，比值随之爆到 **47722**；而同一批数据里**物理行为完全相同**
            # 的段只给出 **1.4**（跨 34000×）。85 轮 `full` **每一次都因此判 0 分**（假红）。
            #
            # ⇒ 换成 `最大单拍速度变化 / 巡航速度`：无量纲、且**参考越平滑它越小**
            #   （那正是我们要的方向）。真机实测它稳定在 **0.0103~0.0212**，
            #   其中 ~0.013 是 S 曲线自身的台阶 —— 是个**有物理含义的常数**，不是噪声。
            #
            # ⚠ 顺带保留上一轮的教训：`|Δv|` 必须在**巡航段之内**取（取整个 chunk 的话
            #   静止段会把统计量拉垮）；且巡航速度为 0 时本段不计。
            seg_v = v[seg_i[0]:seg_i[1] + 1]
            vc = float(np.median(np.abs(seg_v)))
            dv = np.abs(np.diff(seg_v))
            if vc > 0 and dv.size:
                jerk_spikes.append(float(dv.max() / vc))

    if ripples:
        r.set("smooth.vel_ripple_pct", float(np.median(ripples)),
              evidence="各运动段**最长连续巡航段**内 std(v)/|v|，取中位数。"
                       "⚠ v 用位置差分（日志 dq 列在出厂配置下恒 0）")
    else:
        r.skip("smooth.vel_ripple_pct", "切不出巡航段（行程太短或没录到 300Hz 窗口）")

    if tau_sig:
        r.set("smooth.tau_sigma_nm", float(np.median(tau_sig)),
              evidence="巡航段力矩 std，**已按方差相减扣掉量化地板 LSB/√12**")
        lsb_med = I.tau_lsb(n)
        if lsb_med:
            r.set("smooth.tau_vs_quant",
                  float(np.median(tau_sig)) / S.quant_floor_std(lsb_med),
                  evidence=f"归一化到最粗轴的量化地板 {S.quant_floor_std(lsb_med):.4f} N·m")
    else:
        r.skip("smooth.tau_sigma_nm", "没有可用的巡航段")
        r.skip("smooth.tau_vs_quant", "同上")

    if jerk_spikes:
        r.set("smooth.jerk_spike_ratio", float(np.median(jerk_spikes)),
              evidence=f"**巡航段之内** `max|Δv| / 巡航速度` 的中位数（{len(jerk_spikes)} 段可用）"
                       f"—— 即「最大单拍速度变化占巡航速度的比例」，无量纲；"
                       f"v 用位置差分；**单流指标**、无跨流对齐伪影。"
                       f"⚠ 分母**必须是巡航速度**：`median|Δv|` 会随参考变平滑而趋近 f32 噪声"
                       f"（实测低到 3.1e-07），使比值爆到 1e4 量级 —— 那是**判据不稳**，"
                       f"不是真突跳")
    else:
        r.skip("smooth.jerk_spike_ratio", "没有可用的巡航段（或巡航速度为 0）")


# ================================ E. 力矩 =================================== #
def _torque(r: RawResult, trace, segs, identity) -> None:
    tau_max = _tau_max_from_identity(identity)
    if trace.tau.size and tau_max is not None:
        tm = np.asarray(tau_max, float)[: trace.tau.shape[1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(trace.tau) / tm
        r.set("torque.peak_ratio", float(np.nanmax(ratio)),
              evidence=f"全运行 max|τ|/tau_max（逐轴），tau_max 取设备读回值")
        sat = float(np.mean(np.nan_to_num(np.abs(trace.tau) >= 0.98 * tm)))
        r.set("torque.saturation_frac", sat,
              evidence="|τ| ≥ 0.98·tau_max 的采样占比（全运行）")
    else:
        r.skip("torque.peak_ratio", "读不到 tau_max ⇒ 无法归一化")
        r.skip("torque.saturation_frac", "同上")

    # ⚠⚠ 判据：**只在"模型预测得出来"的轴上比**。
    #
    # 真机实测（2026-09-20，零位）：J1 `τ=+0.4396 / G=+0.0000`、J3 `τ=+0.3624 / G=−0.0002`
    # —— 偏差算出来 **100%**。但**那不是模型误差**：那两根轴在零位与重力同向，G 本来就
    # 是 0，而实测 τ 是**库仑摩擦撑着静平衡**的力矩。
    # 旧实现只按 `|τ| > 0.3` 挑轴，把「G 主导」与「摩擦主导」两种**完全不同的工况**
    # 混在一起取中位数 ⇒ 整条指标被从历史分布的中位 **6.3%** 抬到 **35%**
    # （锚点上界 50% ⇒ 判约 20 分）。
    #
    # ⇒ 门：`|G(q_i)| ≥ max(|fc0_i|,|fc1_i|)`（该轴的库仑摩擦）**且** `|τ_i| > 0.3`
    #   —— 前者保证重力项在实测力矩里是**主导**的，这个比值才有意义。
    #
    # ⚠ 样本来自 `trace.meta['g_samples']`（suite 在**稳态**采，瞬态读数会把偏差高估
    #   近一倍 —— 本项目踩过），判据在这里做是因为 `fc0/fc1` 要等 identity 快照。
    samples = trace.meta.get("g_samples")
    fc0, fc1 = identity.get("fc0"), identity.get("fc1")
    if samples and fc0 and fc1:
        errs, n_skip = [], 0
        for smp in samples:
            g = np.asarray(smp.get("g") or [], float)
            t = np.asarray(smp.get("tau") or [], float)
            for i in range(min(len(g), len(t), len(fc0), len(fc1))):
                fric = max(abs(float(fc0[i])), abs(float(fc1[i])))
                if abs(g[i]) < fric or abs(t[i]) <= 0.3:
                    n_skip += 1
                    continue
                errs.append(abs(t[i] - g[i]) / abs(t[i]))
        if errs:
            r.set("torque.g_match_pct", float(np.median(errs)) * 100.0,
                  evidence=f"稳态位形下 `|τ−G(q)|/|τ|` 的中位数（**{len(errs)}** 个轴-位形读数）；"
                           f"**只在 `|G| ≥ 该轴库仑摩擦` 的轴上比** —— 另 {n_skip} 个读数里"
                           f"重力项不主导（零位时 G 本就是 0、实测 τ 由摩擦撑着），"
                           f"比值无意义、已剔除")
        else:
            r.skip("torque.g_match_pct",
                   f"{len(samples)} 个位形共 {n_skip} 个读数**没有一个**满足"
                   f"「重力项主导」（`|G| ≥ 库仑摩擦`）⇒ 这些位形判不了模型可信度")
    elif samples:
        r.skip("torque.g_match_pct", "读不到 `fc0/fc1`（库仑摩擦）⇒ 无法判「重力项是否主导」")
    else:
        r.skip("torque.g_match_pct", "suite 未采到静态 G 对照（需动力学模型可用且已采静止段）")


# =============================== F. 笛卡尔 ================================== #
def _cart_pieces(trace, segs):
    """每个笛卡尔段**自己的** TCP 采样（按 100Hz 时间轴切窗）。"""
    out = []
    for s in segs:
        if s.kind not in ("cart_line", "cart_arc"):
            continue
        m = (trace.tcp_t >= s.t0) & (trace.tcp_t <= s.t1)
        if int(m.sum()) >= 5:
            out.append((s, trace.tcp_t[m], trace.tcp_p[m]))
    return out


def _circumcenter(a, b, c):
    """三角形**外心**（3D）。返回 ``(center, 单位法向, 半径)``；近共线 ⇒ ``None``。"""
    a, b, c = (np.asarray(p, float) for p in (a, b, c))

    def w(x, y, z):                       # 顶点 x 的外心权重（重心坐标）
        cr = np.cross(y - x, z - x)
        d = float(np.dot(cr, cr))
        if d < 1e-18:
            return None
        return float(np.dot(z - y, z - y) * np.dot(x - y, x - z) / (2.0 * d))

    wa, wb, wc = w(a, b, c), w(b, c, a), w(c, a, b)
    if wa is None or wb is None or wc is None:
        return None
    center = wa * a + wb * b + wc * c
    n = np.cross(b - a, c - a)
    n = n / float(np.linalg.norm(n))
    return center, n, float(np.linalg.norm(a - center))


def _dev_line(p, start, goal) -> Optional[float]:
    """采样点到「起点→目标」连线的**最大垂距**（m）。"""
    a, b = np.asarray(start, float), np.asarray(goal, float)
    ab = b - a
    L = float(np.linalg.norm(ab))
    if L < 1e-9:
        return None
    d = np.cross(np.tile(ab, (len(p), 1)), p - a) / L
    return float(np.linalg.norm(d, axis=1).max())


def _dev_arc(p, center, normal, radius) -> float:
    """采样点到**圆**的径向偏差（m）—— 先投到圆所在平面，再比半径。"""
    rel = p - np.asarray(center, float)
    inplane = rel - np.outer(rel @ np.asarray(normal, float), np.asarray(normal, float))
    return float(np.abs(np.linalg.norm(inplane, axis=1) - radius).max())


def _cartesian(r: RawResult, trace, segs) -> None:
    """笛卡尔精度 —— **逐段**算，每段只跟**它自己的**理论路径比。

    ⚠⚠ 旧实现把**整条** `trace.tcp_p` 一次性喂进来（直线一次、圆弧一次），于是两次
    笛卡尔动作的采样被拼在一起。2026-09-20 真机实测的后果：直线段的弦长被算成
    **95.1 mm**（真实 56.5 mm）、偏差报成 **20.6 mm ⇒ 判 0 分** ——
    **那是把圆弧段当成了直线的偏离，是假红**。
    旧证据里本来就写着「⚠ 多点共用一份采样，多次笛卡尔段会互相污染」——
    **写明了却照样当分数用，等于没写。**
    """
    line_dev = arc_dev = None
    plans: List[float] = []
    n_seg = 0
    skipped: List[str] = []

    for s, _tt, p in _cart_pieces(trace, segs):
        n_seg += 1
        pu = (s.extra or {}).get("plan_us")
        if pu:
            plans.append(float(pu))
        poses = (s.extra or {}).get("poses") or []
        if s.kind == "cart_line":
            if not poses:
                skipped.append(f"{s.label}(无目标位姿)")
                continue
            dev = _dev_line(p, p[0], np.asarray(poses[-1][:3], float))
            if dev is not None:
                line_dev = dev if line_dev is None else max(line_dev, dev)
        else:
            if len(poses) < 3:
                skipped.append(f"{s.label}(三点不足)")
                continue
            cc = _circumcenter(*[np.asarray(q[:3], float) for q in poses[:3]])
            if cc is None:
                skipped.append(f"{s.label}(三点近共线 ⇒ 定不出圆)")
                continue
            dev = _dev_arc(p, *cc)
            arc_dev = dev if arc_dev is None else max(arc_dev, dev)

    for key, val, why in (
        ("cart.line_dev_mm", line_dev, "本档未跑直线段或未采到 TCP"),
        ("cart.arc_dev_mm", arc_dev, "本档未跑圆弧段或未采到 TCP"),
    ):
        if val is not None:
            r.set(key, val * 1000.0,
                  evidence=f"{n_seg} 段笛卡尔**逐段**比各自的理论路径（不再共用一份采样）")
        else:
            r.skip(key, why + (f"；跳过：{'、'.join(skipped)}" if skipped else ""))

    if plans:
        r.set("cart.plan_us", max(plans),
              evidence=f"各笛卡尔段 `plan_us` 的**最大值**（{len(plans)} 段："
                       + "、".join(f"{v:.0f}" for v in sorted(plans)) + " µs）")
    else:
        r.skip("cart.plan_us", "本档未跑笛卡尔或固件未上报 plan_us")


# =============================== G. 同步 =================================== #
def _sync(r: RawResult, trace, segs) -> None:
    """多轴同步。⛔ **本判据测的是"指令"不是"臂"，而指令级同步是构造保证的。**

    `movej_sync` 的实现是**一条标量 `s: 0→1` 同时驱动全轴** —— core 的 `movej_sync`
    文档原文：「保证是**参考级**同步（参考同拍到达）；**物理跟踪差异仍在**」。
    ⇒ 各轴**参考**速度必然在同一拍归零 ⇒ 下面这个"停止时刻极差"**恒为 0**。

    真机实测：`sync.stop_spread_ms = 0.0`（恰好 0）⇒ **恒判 100 分**。

    ⛔ 一个**构造即为真**的判据没有判别力 —— 这正是本项目反复栽过的那一类
    （"验收不许只靠『定义即真』的判据"）。⇒ 如实报**未实测**，并把观测到的
    **参考级**极差写进理由：它证明「固件同步特性在参考级确实生效」，
    但**不能**当"臂同步得好"的分数。

    ⚠ 为什么不改去测物理同步：唯一含**实测**位置的是 100 Hz 状态帧（**10ms 分辨率**），
    而各轴物理停止时刻的分散在 ms 量级 ⇒ **低于分辨率**。要量它需要更快的实测流，
    本工程的仪表里没有。
    """
    spreads = []
    for s in segs:
        if s.kind != "movej_sync" or s.chunk is None:
            continue
        tl, qr, _d, _t = trace.log_arrays(s.chunk)
        if len(tl) < 30:
            continue
        ax = moving_axes(tl, qr)
        if len(ax) < 2:
            continue
        # ⚠ 判据 = 各轴**速度非零区间的停止**分散（"全轴同乘一个速度系数"已被实测证伪）
        stops = []
        for j in ax:
            v = np.gradient(qr[:, j], I.CTRL_DT.value)
            w = S.motion_window(v)
            if w:
                stops.append(tl[w[1]])
        if len(stops) >= 2:
            spreads.append((max(stops) - min(stops)) * 1000.0)

    ref = (f"**参考级**极差实测 {float(np.median(spreads)):.3f} ms（{len(spreads)} 段）"
           if spreads else "本档未跑 movej_sync")
    r.skip("sync.stop_spread_ms",
           f"⛔ 判据测的是**指令**：`movej_sync` 由一条标量 s 驱动全轴 ⇒ 参考同拍到达是"
           f"**构造保证**（{ref}）⇒ 恒 0、**无判别力**，不给分。"
           f"物理同步测不了：唯一含实测位置的是 100Hz 流（10ms 分辨率），"
           f"低于各轴停止分散的量级")


# =============================== H. 热 ===================================== #
def _thermal(r: RawResult, trace) -> None:
    if trace.t_mos.size:
        r.set("thermal.t_mos_max", float(np.nanmax(trace.t_mos)),
              evidence="全程 MOS 温度峰值（**只在 100Hz 状态帧里，日志无温度**）")
    else:
        r.skip("thermal.t_mos_max", "没有温度数据")
    if trace.t_coil.size:
        r.evidence["thermal._t_coil_max"] = f"{float(np.nanmax(trace.t_coil)):.1f} °C"
    if trace.t_mos.size and trace.duration > 120:
        t = trace.t_status
        y = trace.t_mos.max(axis=1)
        A = np.vstack([t - t.mean(), np.ones_like(t)]).T
        slope, *_ = np.linalg.lstsq(A, y, rcond=None)
        r.set("thermal.rise_per_min", float(slope[0] * 60.0),
              evidence="全程 MOS 最高温对时间的**线性拟合斜率**（需要 ≥2min 数据）")
    else:
        r.skip("thermal.rise_per_min",
               f"时长 {trace.duration:.0f}s < 120s ⇒ 拟不出有意义的热斜率")


# ============================== I. 耐久 ==================================== #
def _endurance(r: RawResult, trace) -> None:
    if trace.seq.size > 5:
        gap, pairs = S.seq_reconcile(trace.seq)
        r.set("endur.seq_gap", gap,
              evidence=f"相邻两帧收到的 `seq` 之差 − 1 的最大值（取模 65536）—— "
                       f"**只量固件丢帧，不掺主机读取节奏**；**可比 {pairs} 对**"
                       + ("（⚠ 0 对 ⇒ 这条判据什么都没证明）" if pairs == 0 else ""))
        r.notes["endur._seq_pairs"] = str(pairs)
    else:
        r.skip("endur.seq_gap", "状态帧太少")
    r.set("endur.reconnects", float(trace.meta.get("reconnects", 0)),
          evidence="运行期间观察到的意外重连次数（期望恒 0）")
    # ⚠ 快照由 suite 的 `drift_ref` 动作在**首尾各记一次**（同一固定位形、多帧均值）。
    #   旧实现只读 `trace.meta['static_drift_mrad']`，而**全仓没有任何写入点** ——
    #   它不是"要跑某个档才有"，是**一条从来没有数据来源的指标**。
    refs = trace.meta.get("drift_refs") or []
    if len(refs) >= 2:
        a0, a1 = refs[0], refs[-1]
        q0 = np.asarray(a0.get("q") or [], float)
        q1 = np.asarray(a1.get("q") or [], float)
        t0 = np.asarray(a0.get("target") or [], float)
        t1 = np.asarray(a1.get("target") or [], float)
        if q0.size and q0.shape == q1.shape == t0.shape == t1.shape:
            d0 = (q0 - t0) * 1000.0
            d1 = (q1 - t1) * 1000.0
            drift = np.abs(d1 - d0)
            j = int(np.argmax(drift))
            gap = float(a1.get("t", 0.0)) - float(a0.get("t", 0.0))
            r.set("endur.static_drift_mrad", float(drift[j]),
                  evidence=f"首（{a0.get('label')}）尾（{a1.get('label')}）**同一点位**的"
                           f"静差**之差**，逐轴取最大 —— 最差 J{j+1}："
                           f"{d0[j]:+.3f} → {d1[j]:+.3f} mrad；两次间隔 {gap:.0f} s。"
                           f"⚠ 它把**热漂 / 参数漂 / 机械松动**三者混在一起 ⇒ 只作提示",
                  note=f"各轴 |Δ静差| mrad：{np.round(drift, 3).tolist()}")
        else:
            r.skip("endur.static_drift_mrad", "首尾快照的轴数/目标不一致 ⇒ 不可比")
    elif refs:
        r.skip("endur.static_drift_mrad",
               f"只记到 {len(refs)} 次同点位快照（需要 2 次才能比）")
    else:
        r.skip("endur.static_drift_mrad", "本档未记到同点位快照（`drift_ref` 动作没跑到）")


# =============================== J. 安全 =================================== #
def _safety(r: RawResult, trace, identity) -> None:
    wd = trace.meta.get("watchdog_latency_s")
    if wd is not None:
        r.set("safe.watchdog_latency_s", float(wd),
              evidence="停止发命令 → 观察到 `flags & 0x2`（core 解成 **WD_TRIPPED**）置位的时延。"
                       "⛔ 按固件拼法搜 `WATCHDOG` **永远搜不到**（历史事故）")
    else:
        # ⚠ 这里**必须把理由带出来**。常规档位跑完就是"未实测"，而
        #   "未观察到跳闸"读起来像"通过了" —— 那正是「测不到」被当成「测出来很好」。
        why = trace.meta.get("watchdog_note") or "本档未做看门狗断连试验"
        r.skip("safe.watchdog_latency_s", why)

    marg = trace.meta.get("min_margin_rad")
    if marg is not None:
        r.set("safe.margin_rad", float(marg),
              evidence=f"全程最小**距锁存线**裕度（= `软限 ± {I.POS_LATCH_BAND.value}`，"
                       f"逐状态帧逐轴取 min）。⚠ 不是「距软限」——固件锁存判定在包络上"
                       f"（`safety_check.c:165`）")
    else:
        r.skip("safe.margin_rad", "未记录限位裕度")

    n_latch = int(trace.meta.get("pos_violation_events", 0))
    r.set("safe.pos_violation_events", float(n_latch),
          evidence="运行期间 `POSITION_VIOLATION` 出现的次数（期望恒 0：该锁存不受 enabled 门控、reset 清不掉）")


# =========================== K. 固件计算性能 ================================ #
def _compute(r: RawResult, bench: Optional[dict]) -> None:
    """固件计算性能。⚠ 自检里有**两个不同的量**，真机实测它们差 **10×**，必须分清：

    * `loop_max_kcycle` ← 固件 `loop_k = g_arm.loop_max_cyc / 1000`
      （`litearm.h:307` 注释「历史最长单拍耗时」）—— **真实整拍的峰值**，指标取它；
    * `LOOP1` ← 自检里**禁中断单次**测的控制步计算量（`kin_runner.c:415-418`）
      —— 只当旁证、不进分：它把中断全关了，恒**小于**真实整拍
      （真机：`LOOP1` 20.06 kcycle vs `loop_k` **199** kcycle）。
    """
    budget = 480000.0 / 300.0                      # 300Hz @480MHz ⇒ 1600 kcycle
    if not bench:
        r.skip("compute.loop_max_kcycle", "本档未跑固件自检 0x49")
        r.skip("compute.overruns", "同上")
        return

    if bench.get("loop_max_kcycle") is None:
        r.skip("compute.loop_max_kcycle",
               "自检只回了 1 帧（LINK 段缺失）⇒ 控制拍峰值**读不到**，判未实测")
    else:
        iso = bench.get("loop1_isolated_kcycle")
        r.set("compute.loop_max_kcycle", bench["loop_max_kcycle"],
              evidence=f"固件 `loop_k`（历史最长单拍耗时 / 1000）；"
                       f"控制拍预算 = 480MHz/300Hz = {budget:.0f} kcycle"
                       + (f"；旁证 `LOOP1`（禁中断单次）={iso:.1f} kcycle" if iso else ""))

    if bench.get("overruns") is None:
        r.skip("compute.overruns", "同上（LINK 段缺失）")
    else:
        r.set("compute.overruns", bench["overruns"],
              evidence="超预算拍数（期望恒 0）；非 0 即控制周期被打破")

    if bench.get("_frames", 0) < 2:
        r.notes["compute._frames"] = (
            "⚠ 自检只收到 1 帧（LINK 段缺失）⇒ 控制拍峰值与链路计数**全部未实测** —— "
            "这是「要不要填 0」的问题，答案是**不填**（填 0 会把「读不到」伪装成「很好」）")


# =============================== L. 链路 =================================== #
def _link(r: RawResult, trace, identity) -> None:
    if len(trace.t_status) > 10:
        dur = float(trace.t_status[-1] - trace.t_status[0])
        r.set("link.status_hz", (len(trace.t_status) - 1) / dur if dur > 0 else None,
              evidence=f"按到达时间戳自算的状态帧率（协议固定 {I.STATUS_HZ.value:.0f}Hz）")
    else:
        r.skip("link.status_hz", "状态帧太少")

    # ⚠⚠ **两个计数的来源不同，别写成一样**（旧实现把两者都注成"core 的累计计数"）：
    #    * `crc_errors` 是**固件**的 —— "上位机发出的帧被固件丢弃"；
    #    * `foreign_frames` / `unexpected_frames` 是 **core** 的 —— 读循环丢掉的帧。
    #    来源写错会让人去错的地方排查。
    others = []
    for k, nm in (("reply_dropped", "固件应答 FIFO 队满丢弃"),
                  ("can_tx_fail", "CAN 发送失败"),
                  ("rx_fifo_lost", "RX FIFO 溢出（= 静默丢反馈，可能整臂 EMERGENCY）"),
                  ("unexpected_frames", "core 未识别 id 的帧")):
        if trace.meta.get(k) is not None:
            others.append(f"{nm} {trace.meta[k]}")
    tail = ("。本次会话的其余链路计数：" + "、".join(others)) if others else ""

    for key, meta, src in (("link.crc_errors", "crc_errors", "**固件**"),
                           ("link.foreign_frames", "foreign_frames", "**core**")):
        v = trace.meta.get(meta)
        if v is not None:
            r.set(key, float(v), evidence=f"{src}的累计计数 `{meta}`（期望恒 0）" + tail)
        else:
            r.skip(key, f"未记录该计数（{src}侧没取到 ⇒ 判未实测，不填 0）")


# --------------------------------------------------------------------------- #
def _limits_from_identity(identity, n):
    jp = identity.get("joint_params") or []
    if len(jp) < n:
        return None
    return (np.asarray([p["q_min"] for p in jp[:n]], float),
            np.asarray([p["q_max"] for p in jp[:n]], float))


def _kp_hold_from_identity(identity):
    jp = identity.get("joint_params") or []
    gain = identity.get("hold_kp_gain")
    if not jp or gain is None:
        return None
    return [I.kp_hold(p["kp"], float(gain)) for p in jp]


def _tau_max_from_identity(identity):
    jp = identity.get("joint_params") or []
    return [p["tau_max"] for p in jp] if jp else None
