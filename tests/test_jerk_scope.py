"""`smooth.jerk_spike_ratio` —— 取值域与**分母**都必须选对。

这条指标错过两次，两次都是"判据在拿一个不稳定的东西当尺度"：

① **取值域**：`|Δv|` 要在**巡航段之内**取。取整个 chunk 的话，chunk 里大部分拍是
   静止段（Δv≡0）⇒ 统计量被拉垮 ⇒ 恒判未实测。

② **分母**（6 小时浸泡才发现）：旧口径用 `max|Δv| / median|Δv|`，而 `median|Δv|`
   的**量级由参考平滑到什么程度决定** —— 参考越平滑它越接近 **f32 舍入噪声**。
   真机实测它低到 **3.143e-07**，比值爆到 **47722**；而同一批数据里**物理行为完全
   相同**的段只给 **1.4**（跨 34000×）。85 轮 `full` **每一次都因此判 0 分（假红）**。

   ⇒ 换成分母 = **巡航速度**：`max|Δv| / |v|_cruise`，即「最大单拍速度变化占巡航
   速度的比例」。参考越平滑它**越小**（正确方向），真机实测稳定在 0.0103~0.0212。
"""

import numpy as np
import pytest

from litearm_benchmark import instruments as I
from litearm_benchmark.capture import LogChunk, Trace
from litearm_benchmark.metrics import signals as S
from litearm_benchmark.metrics.analysis import RawResult, Segment, _smoothness

N = 7
DT = I.CTRL_DT.value


def _ramp_trace(*, spike: bool, n: int = 300):
    """一条 chunk：静止 → 巡航 → 静止。

    巡航段由**逐拍累加**得到，并按 `float32` 存 —— 这**忠实复现**了真机上
    `median|Δv|` 落进 f32 舍入噪声的机制（那正是旧口径爆掉的原因）。
    `spike=True` 时在某一拍让速度**真的**突跳一次。
    """
    v = 1.0                                   # rad/s 的巡航速度
    pos = np.float32(0.0)
    q = np.zeros((n, N), dtype=np.float32)
    for k in range(n):
        step = np.float32(0.0)
        if 80 <= k < 220:
            step = np.float32(v * DT)
            if spike and k == 150:
                step = np.float32(step + 0.30 * DT)   # 一拍内速度突跳 0.30 rad/s
        pos = np.float32(pos + step)
        q[k, 0] = pos
    tr = Trace()
    tr.chunks = [LogChunk(tick=np.arange(n, dtype=np.int64), q_ref=q.astype(float),
                          dq_ref=np.zeros((n, N)), tau=np.zeros((n, N)), label="c0")]
    tr.q = np.asarray([[0.0] * N])
    tr.t_status = np.zeros(1)
    seg = Segment("ramp", "movej", 0.0, 1.0, target=[0.0] * N, chunk=0, travel_rad=0.5)
    return tr, [seg]


def _run(**kw):
    tr, segs = _ramp_trace(**kw)
    r = RawResult()
    _smoothness(r, tr, segs, identity={}, excited=True)
    return r, tr


def _old_ratio(tr) -> float:
    """旧口径 `max|Δv| / median|Δv|`（在巡航段之内）—— 用来做负控。

    ``median|Δv|`` 恰为 0 时返回 ``inf``（那本身也是"分母退化"的极端形态）。
    """
    _t, qr, _d, _tau = tr.log_arrays(0)
    v = np.gradient(qr[:, 0], DT)
    si = S.cruise_segment(v)
    dv = np.abs(np.diff(v[si[0]:si[1] + 1]))
    med = float(np.median(dv))
    return float(dv.max() / med) if med > 0 else float("inf")


# --------------------------------------------------------------------------- #
def test_smooth_reference_scores_low_not_exploded():
    """**核心负控**：参考平顺时这条指标必须**很小**（好），而不是爆成 1e4。

    旧口径在**同一份数据**上会给出几万倍 —— 那是 f32 噪声当分母的产物。
    """
    r, tr = _run(spike=False)
    v = r.values.get("smooth.jerk_spike_ratio")
    assert v is not None, f"不该判未实测：{r.notes}"
    assert v < 0.05, f"平顺参考给出 {v} —— 分母又落到噪声上了"

    old = _old_ratio(tr)
    assert old > 10.0, f"旧口径本该爆掉（实测真机到过 47722），这里只给了 {old:.1f}"


def test_a_real_one_tick_jump_shows_up():
    """**正控**：真有一拍速度突跳 ⇒ 必须明显变大。否则这条指标没有判别力。"""
    r_ok, _ = _run(spike=False)
    r_bad, _ = _run(spike=True)
    a = r_ok.values["smooth.jerk_spike_ratio"]
    b = r_bad.values["smooth.jerk_spike_ratio"]
    assert b > a * 5, f"突跳没被区分出来：平顺 {a:.4f} vs 突跳 {b:.4f}"


def test_perfectly_linear_cruise_is_zero_not_undefined():
    """**语义**：巡航段严格线性 ⇒ 单拍速度变化恒 0 ⇒ 比值 = **0**（满分）。

    ⚠ 旧口径在这里分母恰为 0 ⇒ **无定义**；新口径分母是巡航速度（>0）⇒ 有定义、
    且答案是"完美"。**「0」与「无定义」在这一版被分开了。**
    """
    tr, segs = _ramp_trace(spike=False)
    _t, qr, _d, _tau = tr.log_arrays(0)
    qr = np.round(qr[:, 0], 9)                      # 抹掉 f32 抖动 ⇒ 严格线性
    qr = qr - qr[0]
    tr2 = Trace()
    tr2.chunks = [LogChunk(tick=np.arange(len(qr), dtype=np.int64),
                           q_ref=np.stack([qr] + [np.zeros_like(qr)] * (N - 1), axis=1),
                           dq_ref=np.zeros((len(qr), N)), tau=np.zeros((len(qr), N)),
                           label="c0")]
    tr2.q = np.asarray([[0.0] * N]); tr2.t_status = np.zeros(1)
    r = RawResult()
    _smoothness(r, tr2, segs, identity={}, excited=True)
    v = r.values.get("smooth.jerk_spike_ratio")
    assert v is not None and v < 1e-4, \
        f"严格线性应当判「完美」（比物理地板 0.013 低几个量级），得到 {v}"
