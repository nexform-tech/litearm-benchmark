"""逐段激励闸 —— "这一段真的动了吗？"

2026-09-20 真机首跑暴露的第三类缺陷：`MIN_EXCITATION_RAD` 原来只有**全程**一版，
于是"重复 N 次"这种动作里，只要臂**没动**（第 2..N 轮的目标恰好等于上一轮落点），
那几次落点的差就只是**传感器噪声**，而它照样被当成"重复定位精度"打分。

真机实测：`gcheck` 的最后一个位形就是 `bench_j2lo`，随后 `小位移@0.3` 的三次"重复"
段长全是 **0.02 s**（一步没动），`static.repeat_ptp_mrad` 拿着 **0.76 mrad 判了 91 分**。

本文件的三层守卫：
1. `segment_excited` 的三态语义（**`None` 绝不许当成"动了"**）；
2. `static.repeat_ptp_mrad` 只认真动过的落点，且没有真重复时**跳过而不是退化**；
3. **位形表守卫**：凡 `reps > 1` 的动作必须给 `via` —— 这条才是能提前抓住它的判据。
"""

import numpy as np
import pytest

from litearm_benchmark.metrics.analysis import (
    MIN_EXCITATION_RAD, RawResult, Segment, _static, _tracking, segment_excited,
)

N = 7


# --------------------------------------------------------------------------- #
# 1. 三态语义
# --------------------------------------------------------------------------- #
def test_segment_excited_is_three_valued():
    """**`None`（没记录行程）绝不许当成"动了"** —— 那是"测不出来"，不是"测出来很好"。"""
    mk = lambda tr: Segment("s", "movej", 0.0, 1.0, target=[0.0] * N, travel_rad=tr)
    assert segment_excited(mk(0.50)) is True
    assert segment_excited(mk(MIN_EXCITATION_RAD)) is True      # 边界取闭区间
    assert segment_excited(mk(0.0)) is False
    assert segment_excited(mk(MIN_EXCITATION_RAD / 10)) is False
    assert segment_excited(mk(None)) is None


# --------------------------------------------------------------------------- #
# 2. 重复性：只认真动过的落点
# --------------------------------------------------------------------------- #
def _holds_trace():
    """4 秒 @100Hz 的假时间轴 + 三个**同目标**静止段。

    A、B 是"走过来"的落点（0.500 / 0.501）；C 是**原地没动**的落点（0.550）。
    正确结果只该由 A、B 得出 = 1 mrad；把 C 算进去会变成 50 mrad。

    ⚠ **着色带必须比窗口宽一圈**：`_rest_hold_windows` 取的是 `t >= t1-0.6`（**含**边界），
    所以拿 `t > 边界` 涂色会漏掉那一个边界样本 —— 第一次写这条测试就踩了：
    C 的均值变成 `(55×0.55+0.5)/56`，负控给出 49.107 而不是 50.0。
    """
    t = np.arange(0.0, 4.0, 0.01)
    q = np.full((len(t), N), 0.5)
    q[t < 1.1] = 0.500
    q[(t >= 1.1) & (t < 2.2)] = 0.501
    q[t >= 2.2] = 0.550

    def rest(t1, travel):
        return Segment("r", "rest", t1 - 0.2, t1, target=[0.5] * N, travel_rad=travel)

    segs = [rest(1.0, 0.50), rest(2.0, 0.50), rest(3.0, 0.0)]   # 第三个没动
    return t, q, segs


def test_repeat_ignores_landings_that_never_moved():
    t, q, segs = _holds_trace()
    tr = type("T", (), {"t_status": t, "q": q, "n_joints": N})()
    r = RawResult()
    _static(r, tr, segs, identity={})          # 无 joint_params/fc ⇒ 地板那项自动跳过

    v = r.values.get("static.repeat_ptp_mrad")
    assert v is not None, f"本该给出 A/B 的极差：{r.notes}"
    assert v == pytest.approx(1.0, abs=0.05), \
        f"极差 {v} mrad —— 没动过的 C 被算进去了（正确值应 ≈1 mrad，含 C 会变 50）"
    assert "真动过" in r.evidence["static.repeat_ptp_mrad"], r.evidence


def test_repeat_gate_has_discriminating_power(monkeypatch):
    """**负控**：把闸门短路掉，同一个输入必须给出被污染的结果（50 mrad）。"""
    import litearm_benchmark.metrics.analysis as A
    monkeypatch.setattr(A, "segment_excited", lambda s: True)   # 假装"都动过"

    t, q, segs = _holds_trace()
    tr = type("T", (), {"t_status": t, "q": q, "n_joints": N})()
    r = RawResult()
    _static(r, tr, segs, identity={})
    assert r.values["static.repeat_ptp_mrad"] == pytest.approx(50.0, abs=0.5), \
        "短路闸门后本该被 C 污染到 50 mrad —— 否则本判据没有判别力"


def test_repeat_is_skipped_not_degraded_when_nothing_repeats():
    """没有真重复 ⇒ **跳过**。⛔ 不许退化成"段内样本极差"去打分 —— 那是**噪声**。"""
    t, q, _ = _holds_trace()
    segs = [Segment("r", "rest", 0.8, 1.0, target=[0.5] * N, travel_rad=0.0)]
    tr = type("T", (), {"t_status": t, "q": q, "n_joints": N})()
    r = RawResult()
    _static(r, tr, segs, identity={})

    assert r.values.get("static.repeat_ptp_mrad") is None
    why = r.notes["static.repeat_ptp_mrad"]
    assert "传感器噪声" in why, why


# --------------------------------------------------------------------------- #
# 3. 跟踪：空动作段不许进对齐
# --------------------------------------------------------------------------- #
def test_tracking_skips_segments_without_travel():
    """段长 0.02 s 的"空动作"里也有 31 个帧 ⇒ 它过得了一般的帧数判据，
    但**不含任何跟踪信息**。逐段激励闸必须把它挡掉，并**说清为什么**。"""
    t = np.arange(0.0, 2.0, 0.01)
    q = np.full((len(t), N), 0.0)
    tr = type("T", (), {"t_status": t, "q": q, "n_joints": N})()
    segs = [Segment("空动作", "movej", 0.0, 0.02, target=[0.0] * N, chunk=0, travel_rad=0.0)]

    r = RawResult()
    _tracking(r, tr, segs)
    assert r.values.get("track.rms_mrad") is None
    why = r.notes["track.rms_mrad"]
    assert "行程不足" in why, why


# --------------------------------------------------------------------------- #
# 4. 位形表守卫 —— **能提前抓住这个 bug 的那条判据**
# --------------------------------------------------------------------------- #
def test_every_repeated_action_defines_a_real_travel():
    """凡 `reps > 1` 的动作，都必须给 `via`，且 `via` 与 `target` 要有真实位移。

    ⚠ 这条判据的价值在于它**跑硬件之前就能红**。当场那三个"重复"之所以是空的，
    就是因为位形表里 `reps` 的目标与上一动作落点重合 —— 而没有任何东西检查过这一点。
    """
    from litearm_benchmark.suites.base import POSES
    from litearm_benchmark.suites import tiers

    bad = []
    for name in tiers.TIERS:
        for a in tiers.actions_of(name):
            if a.kind not in ("movej", "movej_sync") or a.reps <= 1:
                continue
            if not a.via:
                bad.append(f"{name}/{a.label}: reps={a.reps} 但没有 via")
                continue
            tgt, via = np.asarray(POSES[a.target]), np.asarray(POSES[a.via])
            travel = float(np.max(np.abs(tgt - via)))
            if travel < MIN_EXCITATION_RAD:
                bad.append(f"{name}/{a.label}: via↔target 只差 {travel * 1000:.1f} mrad")
    assert not bad, "重复动作的位形表退化了（臂不会动，重复性/跟踪测不到东西）：\n  " + \
                    "\n  ".join(bad)


def test_joint_refuses_a_via_that_is_too_close(tmp_path):
    """**负控**：位形表若退化（via ≈ target），执行器必须**当场拒绝**而不是照跑。"""
    from litearm_benchmark.suites.base import Action, Executor, POSES

    ex = Executor.__new__(Executor)      # `_joint` 在碰 cap/arm **之前**就该抛
    a = Action("movej", "退化的重复", target="bench", via="bench", reps=2)
    with pytest.raises(ValueError, match="退化了"):
        ex._joint(a)
