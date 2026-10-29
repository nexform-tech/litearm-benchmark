"""`dyn.rise_ms` **必须同档比** —— 出处自己就这么写的。

它的 provenance 原文：「它由固件的加速度/加加速度上限决定，**换速度档会成比例变化
⇒ 必须同档比**」。而旧实现**跨所有速度段取一个中位数** ⇒ 量到的主要是
「命令速度有多慢」，不是这台臂的响应。

真机 `full` 档实测（2026-09-21）：speed 0.2→1400 ms、0.3→680、0.4→800、0.5→1110
（**17 段**，耐久循环）、0.6→940、0.8→439、1.0→380 ms。
跨档中位数 **680 ms** 几乎就是被那 17 段主导的；而**最快档**是 **380 ms**。

⚠ 顺带发现它**不单调**（0.4 比 0.5 还快）—— 因为各档**行程不同**，
所以它不是一个单变量曲线。这更说明"跨档取一个数"没有意义。
"""

import numpy as np
import pytest

from litearm_benchmark.metrics.analysis import RawResult, Segment, _dynamic

N = 7


def _trace(specs):
    """`specs` = [(t0, rise_s, speed)] —— 每次都是一个 J1 上的线性斜坡到 0.5 rad。"""
    n = 1400
    t = np.arange(n) * 0.01
    q = np.zeros((n, N))
    segs = []
    for t0, dur, speed in specs:
        m = (t >= t0) & (t < t0 + dur)
        q[m, 0] = 0.5 * (t[m] - t0) / dur
        q[t >= t0 + dur, 0] = 0.5
        segs.append(Segment(f"S{speed:g}", "movej", t0, t0 + dur + 1.0,
                            target=[0.5] + [0.0] * (N - 1), extra={"speed": speed}))
    tr = type("T", (), {"t_status": t, "q": q})()
    return tr, segs


def test_reports_the_fastest_speed_tier_not_a_cross_tier_median():
    """**正控**：两档速度差 4 倍 ⇒ 必须报**最快档**（1.0）那个数。"""
    # 线性斜坡 ⇒ 10%→90% 的时间 = 0.8×斜坡时长
    tr, segs = _trace([(0.0, 2.0, 0.3), (4.0, 0.5, 1.0)])
    r = RawResult()
    _dynamic(r, tr, segs)

    v = r.values["dyn.rise_ms"]
    assert v == pytest.approx(0.8 * 0.5 * 1000, rel=0.05), \
        f"报成了 {v:.0f} ms —— 应当是 speed=1.0 档的 ~400 ms"
    ev = r.evidence["dyn.rise_ms"]
    assert "最快档" in ev and "同档比" in ev
    assert "speed=0.3" in ev and "speed=1" in ev, f"各档分布要写出来：{ev}"


def test_the_cross_tier_median_would_have_been_a_different_number():
    """**负控**：证明"跨档取中位数"确实会给出另一个（误导性的）数。

    这正是这条判据的判别力来源：同一份数据，换取值域就换一个结论。
    """
    tr, segs = _trace([(0.0, 2.0, 0.3), (4.0, 0.5, 1.0)])
    r = RawResult()
    _dynamic(r, tr, segs)

    pooled = float(np.median([0.8 * 2.0 * 1000, 0.8 * 0.5 * 1000]))
    assert pooled == pytest.approx(1000.0, rel=0.01)
    assert abs(r.values["dyn.rise_ms"] - pooled) > 400, \
        "新值必须与跨档中位数明显不同，否则这条判据没起作用"
