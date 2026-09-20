"""信号原语 —— 指标层复用的、**每个都带着一个历史教训**的小函数。

放这里的理由：这些算子在项目历史里各错过一次，集中一处才好一次说清"为什么这么写"。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# 稳健统计
# --------------------------------------------------------------------------- #

def p95(x: np.ndarray) -> float:
    """p95。用百分位而不是 max —— 单点野值不该定义一个指标。"""
    x = np.asarray(x, float)
    return float(np.percentile(x, 95)) if x.size else float("nan")


def robust_sigma(x: np.ndarray) -> float:
    """1.4826×MAD 的稳健 std —— 比 std 抗野值（本项目的信号里野值很多）。"""
    x = np.asarray(x, float)
    if x.size < 2:
        return float("nan")
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def quant_floor_std(lsb: float) -> float:
    """均匀量化本身贡献的 std = LSB/√12。**判「抖动/纹波」前必须先减去它。**

    不扣的后果：J1/J2 的 `tau` LSB = 0.0586 N·m ⇒ 地板 std = 0.0169 N·m，
    比 J5-J7 的真实纹波还大 ⇒ 会把「粗量化的台阶」误判成「力矩在抖」。
    """
    return float(lsb) / 12 ** 0.5


def excess_over_floor(sigma_measured: float, floor: float) -> float:
    """从测得的 std 里**减去**量化地板的贡献（方差相减，结果不为负）。

    ⚠ 不是直接相减：抖动与量化是**独立**的方差源，`σ² = σ_真² + σ_地板²`。
    直接减会把真值系统性低估。
    """
    if not np.isfinite(sigma_measured) or sigma_measured <= 0:
        return 0.0
    v = sigma_measured ** 2 - float(floor) ** 2
    return float(v ** 0.5) if v > 0 else 0.0


# --------------------------------------------------------------------------- #
# 静止抖动 —— ⚠ 必须用**直线拟合残差**，不能从单频分量反推
# --------------------------------------------------------------------------- #

def fit_line_residual_std(t: np.ndarray, x: np.ndarray) -> float:
    """对位置做**直线拟合**，返回残差 std。

    ⚠ 为什么不用 FFT 单频分量反推：本项目首版报告就是那么做的，**低估了 3~4 倍**。
    静止段若还有缓慢漂移（热漂/摩擦爬行），直线拟合会把它扣掉、只留真抖动 ——
    这正是我们要的。
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    if t.size < 3:
        return float("nan")
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, x - x.mean(), rcond=None)
    resid = (x - x.mean()) - A @ coef
    return float(resid.std(ddof=1)) if resid.size > 2 else float("nan")


def dominant_freq_hz(t: np.ndarray, x: np.ndarray, fmin: float = 1.0,
                     fmax: float = 40.0) -> Tuple[float, float]:
    """带内主导频率与它占的功率比。返回 ``(f_hz, ratio)``。

    ⚠ **两种谱不能混用**（本项目记录过）："某单频占比 55~68%" 只对**差分速度谱**成立；
    在**位置残差谱**里那条线只占很小一部分、主体是宽带。调用方必须说明自己给的是哪种信号。
    """
    t = np.asarray(t, float)
    x = np.asarray(x, float) - np.mean(x)
    if t.size < 16:
        return float("nan"), float("nan")
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return float("nan"), float("nan")
    n = x.size
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * w)) ** 2
    freqs = np.fft.rfftfreq(n, dt)
    m = (freqs >= fmin) & (freqs <= fmax)
    if not m.any() or spec[m].sum() <= 0:
        return float("nan"), float("nan")
    i = int(np.argmax(np.where(m, spec, 0)))
    return float(freqs[i]), float(spec[i] / spec[m].sum())


# --------------------------------------------------------------------------- #
# 稳健标尺 —— ⚠ 绝不用 `max|v|` 当阈值基准
# --------------------------------------------------------------------------- #

def robust_peak(x: np.ndarray, pct: float = 98.0) -> float:
    """稳健峰值 = ``|x|`` 的高百分位。

    ⚠ **阈值不许建立在 `max|·|` 上**：一个单拍尖峰（收尾参考台阶、一次野值）会把
    标尺整个抬起来，于是"真巡航段不合格、只有尖峰合格"。本工程的测试就抓到过这个：
    `v[50:130]=1.0` 加一个 `v[195]=1.5` 的尖峰后，按 `0.9·max` 判会把巡航段判丢、
    只挑出那一拍尖峰。
    """
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    return float(np.percentile(np.abs(x), pct))


# --------------------------------------------------------------------------- #
# 运动段切分
# --------------------------------------------------------------------------- #

def motion_window(v: np.ndarray, frac: float = 0.02) -> Optional[Tuple[int, int]]:
    """按速度界定运动段：``|v| > frac·max|v|`` 的**首末下标**。

    ⚠ 别用"位置离开起点超过阈值"来找运动段：**收尾后的保持段**会让位置一直满足条件，
    于是把整段静止也并进来（本项目真实踩过：段长 1070 拍 = 3.57s，而运动只要 0.3s，
    结果 jerk 极值全落在保持段上）。
    """
    v = np.asarray(v, float)
    if v.size < 3:
        return None
    vm = robust_peak(v)
    if not np.isfinite(vm) or vm <= 0:
        return None
    idx = np.where(np.abs(v) > frac * vm)[0]
    return (int(idx[0]), int(idx[-1])) if idx.size >= 2 else None


def cruise_segment(v: np.ndarray, thresh: float = 0.9) -> Optional[Tuple[int, int]]:
    """**最长连续**超阈段（``|v| > thresh·max|v|``）—— 取巡航段用。

    ⚠ 必须取"最长**连续**段"，不能取首末超阈下标：收尾那一拍的参考尖峰会让首末跨到
    减速段去，把纹波虚高过 **12 倍**（本项目实测）。
    """
    v = np.asarray(v, float)
    if v.size < 3:
        return None
    vm = robust_peak(v)
    if not np.isfinite(vm) or vm <= 0:
        return None
    mask = np.abs(v) > thresh * vm
    best = cur = None
    start = 0
    for i, f in enumerate(mask):
        if f:
            if cur is None:
                start = i
            cur = (start, i)
            if best is None or (cur[1] - cur[0]) > (best[1] - best[0]):
                best = cur
        else:
            cur = None
    return best


def ripple_pct(x: np.ndarray, lo: int, hi: int) -> float:
    """区间内的相对纹波 = std / |mean| × 100%。"""
    seg = np.asarray(x, float)[lo:hi + 1]
    if seg.size < 4:
        return float("nan")
    m = abs(float(seg.mean()))
    if m <= 0:
        return float("nan")
    return float(seg.std(ddof=1) / m * 100.0)


# --------------------------------------------------------------------------- #
# 到位指标
# --------------------------------------------------------------------------- #

def overshoot_rebound(q: np.ndarray, target: float, direction: float
                      ) -> Tuple[float, float]:
    """返回 ``(越峰, 回摆)``（对**指令目标**取，不是对第一次落点）。

    * 越峰 = 落点越过目标的最大量；
    * 回摆 = 越峰之后**往回收**的幅度，**恒 ≥ 0**。

    ⚠ 历史事故：曾把「交接残余 − 停稳偏差」这个**代数量**当成回摆，符号是反的、
    而且它只在"交接残余恰好是峰值"时才等于回摆 —— **不是恒等式**。
    """
    q = np.asarray(q, float)
    if q.size < 2:
        return float("nan"), float("nan")
    sg = 1.0 if direction >= 0 else -1.0
    ov = (q - target) * sg
    peak = float(ov.max())
    i = int(np.argmax(ov))
    rebound = float(peak - ov[i:].min()) if i < ov.size else 0.0
    return peak, max(rebound, 0.0)


def settle_time(t: np.ndarray, q: np.ndarray, target: float, band: float,
                t0: float) -> float:
    """从 ``t0`` 起，进入并**保持**在 ``|q−target| ≤ band`` 内所需的时长。"""
    t = np.asarray(t, float); q = np.asarray(q, float)
    m = t >= t0
    if not m.any():
        return float("nan")
    tt, qq = t[m] - t0, q[m]
    inside = np.abs(qq - target) <= band
    # 从**末尾**往前找第一个"出带"点 ⇒ 之后的都一直带内
    last_out = np.where(~inside)[0]
    if last_out.size == 0:
        return float(tt[0]) if tt.size else float("nan")
    k = int(last_out[-1])
    return float(tt[k + 1]) if k + 1 < tt.size else float("nan")


def rise_time(t: np.ndarray, q: np.ndarray, q_from: float, q_to: float,
              t0: float, frac: float = 0.9) -> float:
    """从 ``t0`` 起，走完 ``frac`` 行程所需的时长（10%→90% 的常用代理）。"""
    t = np.asarray(t, float); q = np.asarray(q, float)
    m = t >= t0
    if m.sum() < 3:
        return float("nan")
    tt, qq = t[m] - t0, q[m]
    span = q_to - q_from
    if abs(span) <= 0:
        return float("nan")
    prog = (qq - q_from) / span
    lo = np.where(prog >= 0.10)[0]
    hi = np.where(prog >= frac)[0]
    if not lo.size or not hi.size:
        return float("nan")
    return float(tt[int(hi[0])] - tt[int(lo[0])])


# --------------------------------------------------------------------------- #
# 状态帧序号对账
# --------------------------------------------------------------------------- #

def seq_reconcile(seq: Sequence[int], t: Sequence[float], nominal_hz: float = 100.0,
                  tol_frames: float = 2.0):
    """用 16-bit `seq` 与步长对账，返回 ``(最大缺口帧数, 可比对数)``。

    ⚠ **必须同时报「可比对数」**：只有 0 对时不能说"没有缺口" —— 那与"没有数据"是
    同一句话（本项目记录过"计数器恒 0"被读成"没出错"的教训）。

    `seq` 是 16-bit 计数器会回绕，故增量一律取 ``(b−a) mod 65536``。
    """
    seq = list(int(s) for s in seq)
    t = list(float(x) for x in t)
    worst, pairs = 0.0, 0
    for i in range(1, len(seq)):
        dt = t[i] - t[i - 1]
        if dt <= 0 or dt > 1.0:        # 间隔过大的那对不可比（可能重连）
            continue
        expect = dt * nominal_hz
        got = (seq[i] - seq[i - 1]) % 65536
        gap = abs(got - expect)
        if gap > 65536 / 2:            # 反向回绕
            gap = 65536 - gap
        worst = max(worst, gap)
        pairs += 1
    return float(worst), pairs
