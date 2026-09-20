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
            *, tcp_line: Optional[dict] = None, tcp_arc: Optional[dict] = None,
            bench: Optional[dict] = None) -> RawResult:
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
        _cartesian(r, tcp_line, tcp_arc)
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

    # 重复性：**同一目标**的多段之间取极差；没有重复目标就退化为「同段内各样本的极差」
    by_tgt: Dict[tuple, List[np.ndarray]] = {}
    for s, _t, qw in holds:
        if not s.target:
            continue
        key = tuple(round(v, 4) for v in np.asarray(s.target, float))
        by_tgt.setdefault(key, []).append(qw.mean(axis=0))
    ptp = 0.0
    detail = ""
    if any(len(v) >= 2 for v in by_tgt.values()):
        for key, arrs in by_tgt.items():
            if len(arrs) < 2:
                continue
            ptp = max(ptp, float(np.ptp(np.stack(arrs), axis=0).max() * 1000.0))
        detail = "同一目标多轮之间的极差"
    else:
        for _s, _t, qw in holds:
            ptp = max(ptp, float(np.ptp(qw, axis=0).max() * 1000.0))
        detail = "无重复目标 ⇒ 退化为段内样本极差（**不是重复定位精度**，报告里会标注）"
    r.set("static.repeat_ptp_mrad", ptp, evidence=detail)

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
    for s in segs:
        if s.chunk is None or s.kind not in ("movej", "movej_sync", "move_l", "move_c"):
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
        for k in keys:
            r.skip(k, "没有可对齐的运动段（需要 300Hz 日志窗口 + ≥30 个状态帧）")
        return

    good = [x for x in results if x[1].ok]
    r.set("track.rms_mrad",
          (max(x[1].rms for x in good) * 1000.0) if good else None,
          evidence=f"逐段互相关对齐后取**最差**一段；{len(good)}/{len(results)} 段自证达标")
    r.set("track.peak_mrad",
          (max(x[1].peak for x in good) * 1000.0) if good else None,
          evidence="同上的残差峰值")
    r.set("track.align_ratio",
          min(x[1].ratio for x in results) if results else None,
          evidence="**自证量**取最差（1.0≈噪声，真结构 3~6×）")
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
    ovs, rbs, sts, rises = [], [], [], []
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
            rises.append(rt * 1000.0)

    def put(key, arr, ev):
        if arr:
            r.set(key, float(np.median(arr)), ev)
        else:
            r.skip(key, "没有可用的关节运动段")

    put("dyn.overshoot_mrad", ovs, "各段越峰的中位数（逐段取主运动轴，对**指令目标**取）")
    put("dyn.rebound_mrad", rbs, "各段回摆的中位数（峰值后往回收的幅度，恒 ≥0）")
    put("dyn.settle_s", sts, "各段整定时间中位数（进入 |q−目标| ≤ 10 mrad 并保持）")
    put("dyn.rise_ms", rises, "各段 10%→90% 行程时间中位数")


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
            dv = np.abs(np.diff(v))
            med = float(np.median(dv))
            if med > 0:
                jerk_spikes.append(float(dv.max() / med))

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
              evidence="巡航段 max|Δv| / median|Δv| 的中位数。**单流指标**，无跨流对齐伪影")
    else:
        r.skip("smooth.jerk_spike_ratio", "没有可用的巡航段；或巡航段太抖导致 median|Δv|=0")


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

    g = identity.get("g_match_pct")
    if g is not None:
        r.set("torque.g_match_pct", float(g),
              evidence="稳态多姿态下 |τ_实测 − G(q)| / |τ_实测|，由 suite 在静止段采（**必须用稳态读数**）")
    else:
        r.skip("torque.g_match_pct", "suite 未采到静态 G 对照（需动力学模型可用且已采静止段）")


# =============================== F. 笛卡尔 ================================== #
def _cartesian(r: RawResult, tcp_line: Optional[dict], tcp_arc: Optional[dict]) -> None:
    for key, d, why in (
        ("cart.line_dev_mm", tcp_line, "本档未跑直线段或未采到 TCP"),
        ("cart.arc_dev_mm", tcp_arc, "本档未跑圆弧段或未采到 TCP"),
    ):
        if d and d.get("dev_mm") is not None:
            r.set(key, d["dev_mm"], evidence=d.get("evidence", "TCP 采样 vs 理论路径"))
        else:
            r.skip(key, why)
    if tcp_line and tcp_line.get("plan_us"):
        r.set("cart.plan_us", tcp_line["plan_us"],
              evidence="固件 `CartPlan.plan_us`（规划耗时）")
    else:
        r.skip("cart.plan_us", "本档未跑笛卡尔或固件未上报 plan_us")


# =============================== G. 同步 =================================== #
def _sync(r: RawResult, trace, segs) -> None:
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
    if spreads:
        r.set("sync.stop_spread_ms", float(np.median(spreads)),
              evidence="各轴**速度归零时刻**的极差（判据不是「全轴同乘一个速度系数」）")
    else:
        r.skip("sync.stop_spread_ms", "本档未跑 movej_sync（或运动轴少于 2 个）")


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
        gap, pairs = S.seq_reconcile(trace.seq, trace.t_status, I.STATUS_HZ.value)
        r.set("endur.seq_gap", gap,
              evidence=f"16-bit seq 与步长对账（取模 65536）；**可比 {pairs} 对**"
                       + ("（⚠ 0 对 ⇒ 这条判据什么都没证明）" if pairs == 0 else ""))
        r.notes["endur._seq_pairs"] = str(pairs)
    else:
        r.skip("endur.seq_gap", "状态帧太少")
    r.set("endur.reconnects", float(trace.meta.get("reconnects", 0)),
          evidence="运行期间观察到的意外重连次数（期望恒 0）")
    d = trace.meta.get("static_drift_mrad")
    if d is not None:
        r.set("endur.static_drift_mrad", float(d),
              evidence="长时间运行前后**同一点位**的静差变化（会把热漂/参数漂/松动混在一起）")
    else:
        r.skip("endur.static_drift_mrad", "本档未做首尾同点位对照（需 manual/完整档）")


# =============================== J. 安全 =================================== #
def _safety(r: RawResult, trace, identity) -> None:
    wd = trace.meta.get("watchdog_latency_s")
    if wd is not None:
        r.set("safe.watchdog_latency_s", float(wd),
              evidence="停止发命令 → 观察到 `flags & 0x2`（core 解成 **WD_TRIPPED**）置位的时延。"
                       "⛔ 按固件拼法搜 `WATCHDOG` **永远搜不到**（历史事故）")
    else:
        r.skip("safe.watchdog_latency_s", "本档未做看门狗断连试验")

    marg = trace.meta.get("min_margin_rad")
    if marg is not None:
        r.set("safe.margin_rad", float(marg),
              evidence="全程最小距软限裕度（逐状态帧逐轴取 min）")
    else:
        r.skip("safe.margin_rad", "未记录限位裕度")

    n_latch = int(trace.meta.get("pos_violation_events", 0))
    r.set("safe.pos_violation_events", float(n_latch),
          evidence="运行期间 `POSITION_VIOLATION` 出现的次数（期望恒 0：该锁存不受 enabled 门控、reset 清不掉）")


# =========================== K. 固件计算性能 ================================ #
def _compute(r: RawResult, bench: Optional[dict]) -> None:
    if not bench:
        r.skip("compute.loop_max_kcycle", "本档未跑固件自检 0x49")
        r.skip("compute.overruns", "同上")
        return
    r.set("compute.loop_max_kcycle", bench.get("loop_max_kcycle"),
          evidence="固件自检 `LOOP1` 段的 DWT cycle 峰值 / 1000。"
                   f"控制拍预算 = 480MHz/300Hz = {480000/300:.0f} kcycle")
    r.set("compute.overruns", bench.get("overruns"),
          evidence="超预算拍数（期望恒 0）")
    if bench.get("_frames", 0) < 2:
        r.notes["compute._frames"] = ("⚠ 自检只收到 1 帧 ⇒ LINK 便捷出口会**静默返回 0**，"
                                      "上面的数可能不全")


# =============================== L. 链路 =================================== #
def _link(r: RawResult, trace, identity) -> None:
    if len(trace.t_status) > 10:
        dur = float(trace.t_status[-1] - trace.t_status[0])
        r.set("link.status_hz", (len(trace.t_status) - 1) / dur if dur > 0 else None,
              evidence="按到达时间戳自算的状态帧率（协议固定 100Hz）")
    else:
        r.skip("link.status_hz", "状态帧太少")
    for key, meta in (("link.crc_errors", "crc_errors"), ("link.foreign_frames", "foreign_frames")):
        v = trace.meta.get(meta)
        if v is not None:
            r.set(key, float(v), evidence=f"core 的累计计数 `{meta}`（期望恒 0）")
        else:
            r.skip(key, "未记录该计数")


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
