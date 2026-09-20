"""`smooth.jerk_spike_ratio` 的 `|Δv|` **必须在巡航段之内取**。

2026-09-20 真机：这条指标每档都判未实测，理由是「巡航段太抖导致 median|Δv|=0」。
查下去发现根因是**取值域错了** —— 旧实现在**整个 chunk** 上取 `np.diff(v)`，
而一个动作 chunk 里**大部分拍是静止段**（那一段 v≡0 ⇒ Δv≡0）⇒ 中位数被它们拉到 0
⇒ `med > 0` **恒假**。

真机实测对照（`standard` 档）：

| chunk | 速度档 | 整段 med\\|Δv\\| | **巡航段内** med\\|Δv\\| | max/med |
|---|---|---|---|---|
| 3 | 1.0 | **0.000000** | 0.008002 | **3.33** |
| 5 | 0.5 | 0.000000 | **0.000000** | 分母无定义 |

⇒ 修好后 1.0 档**出得来数**；而 0.5 档**仍**无定义 —— 那是判据的分辨力边界
（参考在巡航段严格线性时，分母恰为 0），必须与"取值域错"分开对待：
前者是**如实剔除**，后者是**实现 bug**。
"""

import numpy as np
import pytest

from litearm_benchmark import instruments as I
from litearm_benchmark.capture import LogChunk, Trace
from litearm_benchmark.metrics import signals as S
from litearm_benchmark.metrics.analysis import RawResult, Segment, _smoothness

N = 7


def _trace_with_one_ramp(*, ramp_variation: float):
    """一条 chunk：静止 → 巡航（带微小逐拍变化）→ 静止。

    `ramp_variation=0` ⇒ 参考在巡航段**严格线性**（Δv≡0，分母无定义）；
    给个小量 ⇒ 巡航段内 Δv≠0（可测）。
    """
    n = 300
    q = np.zeros((n, N))
    v = 0.30                                     # rad/s 的巡航速度
    dt = I.CTRL_DT.value
    pos = 0.0
    for k in range(n):
        if 80 <= k < 220:
            # ⚠ 调制**不能用逐拍 `±e` 交替** —— 那在 `np.gradient` 的中心差分里
            #   会正好抵消（`step_k + step_{k+1} = 2v·dt` 恒定），速度反而完全不变，
            #   于是"巡航段内 Δv≠0"这个前提根本不成立（本测试第一版就踩了这个）。
            step = v * dt * (1.0 + ramp_variation * np.sin(k * 0.9))
            pos += step
            q[k, 0] = pos
        else:
            q[k, 0] = pos                        # 静止段：位置不变 ⇒ v≡0 ⇒ Δv≡0
    tr = Trace()
    tr.chunks = [LogChunk(tick=np.arange(n, dtype=np.int64), q_ref=q,
                          dq_ref=np.zeros((n, N)), tau=np.zeros((n, N)), label="c0")]
    tr.q = np.asarray([[0.0] * N])               # 静止段用；随便给一行
    tr.t_status = np.zeros(1)
    seg = Segment("ramp", "movej", 0.0, 1.0, target=[0.0] * N, chunk=0, travel_rad=0.5)
    return tr, [seg]


def _compute(variation):
    tr, segs = _trace_with_one_ramp(ramp_variation=variation)
    r = RawResult()
    _smoothness(r, tr, segs, identity={}, excited=True)
    return r, tr


def test_jerk_is_taken_within_the_cruise_segment_not_the_whole_chunk():
    """**正控**：巡航段内 Δv≠0 时，这条指标**必须出得来数**。

    旧实现（整段取中位数）在同一个输入上会判未实测 —— 见下一条负控。
    """
    r, _ = _compute(variation=1e-3)
    v = r.values.get("smooth.jerk_spike_ratio")
    assert v is not None, f"仍判未实测：{r.notes.get('smooth.jerk_spike_ratio')}"
    assert np.isfinite(v) and v >= 1.0, f"比值应当 ≥1，得到 {v}"
    assert "巡航段之内" in r.evidence["smooth.jerk_spike_ratio"]


def test_the_whole_chunk_median_would_have_been_zero():
    """**负控**：证明旧取值域确实会得到 median|Δv|=0 ⇒ 恒判未实测。

    这条是上一条的判别力来源：同一份数据，换取值域就从"有数"变"没数"。
    """
    tr, _ = _trace_with_one_ramp(ramp_variation=1e-3)
    _t, qr, _d, _tau = tr.log_arrays(0)
    v = np.gradient(qr[:, 0], I.CTRL_DT.value)

    whole = float(np.median(np.abs(np.diff(v))))
    si = S.cruise_segment(v)
    assert si is not None, "没切出巡航段，测试前提不成立"
    inside = float(np.median(np.abs(np.diff(v[si[0]:si[1] + 1]))))

    assert whole == 0.0, f"整段中位数本该是 0（静止段占多数），得到 {whole}"
    assert inside > 0.0, "巡航段内中位数本该 >0"


def test_perfectly_linear_cruise_is_reported_as_undefined_not_as_smooth():
    """**分辨力边界**：参考在巡航段**严格线性**时分母恰为 0 ⇒ **无定义**。

    ⚠ 必须与"很平滑"分开 —— 分母为 0 时比值**不存在**，
    报 0 或满分都是把"测不了"说成"很好"。
    """
    r, _ = _compute(variation=0.0)
    assert r.values.get("smooth.jerk_spike_ratio") is None
    why = r.notes["smooth.jerk_spike_ratio"]
    assert "无定义" in why and "很平滑" in why
