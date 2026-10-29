"""采集层的时间轴 —— `t_status` 必须是**真实**的 `time.monotonic()`。

这一整个文件是为 2026-09-20 真机首跑写的：那次 `static.err_max_mrad` 报出
**798.769 mrad**，而臂实际走得很准。根因是 `DualCapture._flush` 在拼接帧缓冲时
把 `new_t` 平移成"紧接上一段"，**抹掉了两次 flush 之间的真实间隔** ——
`t_status` 成了被压缩的假时间轴，而 `Segment.t0/t1` 来自 `time.monotonic()`。

两个不同源的时间直接比大小 ⇒ 每个静止段选到的是**别的动作**的片段。
0.8 rad 恰好是相邻两个目标之差，就是这么来的。

⚠ 本文件的判据：**跨 flush 的真实间隔必须原样保留**。旧实现在这条上必红。
"""

import numpy as np
import pytest
from types import SimpleNamespace

from litearm_benchmark.capture import DualCapture, Trace

N = 7


def _state(seed: float):
    """一个够 `_flush` 用的假状态帧（构造只求字段齐，数值不重要）。"""
    j = lambda: SimpleNamespace(q=seed, dq=0.0, tau=0.0, t_mos=30.0, t_coil=28.0, err=0)
    return SimpleNamespace(joints=[j() for _ in range(N)], flags=0, seq=int(seed), mode=0)


def _cap() -> DualCapture:
    cap = DualCapture.__new__(DualCapture)          # 只测 _flush，不碰真机/桩
    cap.trace = Trace()
    return cap


def _feed(cap: DualCapture, times) -> None:
    cap._rec_t = list(times)
    cap._rec = [_state(t) for t in times]
    cap._flush()


def test_flush_preserves_the_gap_between_flushes():
    """**负控**：两次 flush 之间空了 3 秒没帧，那 3 秒必须留在时间轴上。

    旧实现（`new_t - new_t[0] + t[-1] + 1e-6`）会把这段间隔压成 ~0 ⇒ 本断言红。
    """
    cap = _cap()
    _feed(cap, [100.00, 100.01])
    _feed(cap, [103.00, 103.01])          # 中间 2.99 s 无帧

    t = cap.trace.t_status
    assert len(t) == 4
    assert t[2] - t[1] == pytest.approx(103.00 - 100.01, abs=1e-6), \
        f"跨 flush 的间隔被抹掉了：t={t.tolist()}"


def test_flush_timeline_is_monotonic_and_keeps_absolute_values():
    """时间轴单调，且**绝对值就是喂进去的那个**（不许被平移过）。"""
    cap = _cap()
    _feed(cap, [100.00, 100.01])
    _feed(cap, [103.00, 103.01])
    t = cap.trace.t_status
    assert np.all(np.diff(t) > 0), f"时间轴非单调：{t.tolist()}"
    assert t[0] == pytest.approx(100.00)
    assert t[-1] == pytest.approx(103.01)


def test_flush_refuses_a_timeline_that_goes_backwards():
    """时钟倒退 ⇒ **立刻抛**，不许静默产出一条坏时间轴。

    静默的坏时间轴是本工程最贵的一种假数据：它不会让任何东西变红，
    只会让每个指标都算在错误的时刻上。
    """
    cap = _cap()
    _feed(cap, [100.00, 100.01])
    with pytest.raises(RuntimeError, match="倒退"):
        _feed(cap, [99.00, 99.01])


def test_hold_samples_instead_of_sleeping():
    """`Executor._hold` 必须**边保持边读** —— 裸 `sleep` 期间一帧都收不到。"""
    from litearm_benchmark.suites.base import Executor

    calls = []

    class _Arm:
        def get_state(self, refresh=False):
            calls.append(refresh)
            return SimpleNamespace(value=None)

    ex = Executor.__new__(Executor)
    ex.arm = _Arm()
    import time as _t
    t0 = _t.monotonic()
    ex._hold(0.25)
    dt = _t.monotonic() - t0

    assert len(calls) >= 10, f"0.25s 只读了 {len(calls)} 次 ⇒ 采样不足，静止段会缺样本"
    assert all(calls), "必须用 refresh=True 才会真的去读串口"
    assert dt >= 0.25
    assert dt < 1.0, f"保持 0.25s 却花了 {dt:.2f}s"
