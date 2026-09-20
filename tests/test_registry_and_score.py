"""指标目录与打分层的判据 —— **每条判据都配一条负控**。

家族惯例：判据必须"在该红时会红"，否则绿什么也证明不了
（见 `litearm-server/tests/test_rpc_call_sites.py` 的同名模式）。
"""

import pytest

from litearm_benchmark.metrics.analysis import RawResult
from litearm_benchmark.metrics.registry import (
    ALL_METRICS, BY_KEY, DIMENSION_ORDER, Direction, MetricSpec, audit_curves,
)
from litearm_benchmark.score import score


# --------------------------------------------------------------------------- #
# 1. 目录本身合规
# --------------------------------------------------------------------------- #
def test_curves_are_wellformed():
    """锚点升序 / 分数在 0~100 / 单调方向与 direction 一致 —— 一个都不能错。

    这三类错本项目都犯过：错序会让插值算出非单调的分数，方向写反会让"越差越高分"。
    """
    bad = audit_curves()
    assert not bad, "指标曲线不合规：\n  " + "\n  ".join(bad)


def test_audit_has_discriminating_power():
    """**负控**：故意造三条坏曲线，`audit_curves` 必须分别抓到。

    没有这条，`test_curves_are_wellformed` 在实现被改坏时会假绿。
    """
    import litearm_benchmark.metrics.registry as R
    good = R.ALL_METRICS
    broken = [
        # ① 锚点没升序
        MetricSpec("x.bad1", "d", "u", Direction.LOWER_BETTER,
                   ((10.0, 100), (1.0, 0)), "test"),
        # ② 分数越界
        MetricSpec("x.bad2", "d", "u", Direction.LOWER_BETTER,
                   ((0.0, 150), (1.0, 0)), "test"),
        # ③ 声明 lower-better 却给大值高分
        MetricSpec("x.bad3", "d", "u", Direction.LOWER_BETTER,
                   ((0.0, 0), (1.0, 100)), "test"),
    ]
    try:
        R.ALL_METRICS = tuple(good) + tuple(broken)
        bad = R.audit_curves()
        keys = " ".join(bad)
        assert "x.bad1" in keys and "x.bad2" in keys and "x.bad3" in keys, \
            f"自检漏报：{bad}"
    finally:
        R.ALL_METRICS = good
    assert not audit_curves(), "恢复后应重新合规"


def test_every_dimension_has_metrics_and_weight():
    from litearm_benchmark.metrics.registry import DIMENSION_WEIGHTS
    dims = {m.dimension for m in ALL_METRICS}
    assert dims <= set(DIMENSION_ORDER), f"有维度不在 DIMENSION_ORDER 里：{dims - set(DIMENSION_ORDER)}"
    missing = [d for d in dims if d not in DIMENSION_WEIGHTS]
    assert not missing, f"这些维度没有权重：{missing}"


# --------------------------------------------------------------------------- #
# 2. 打分语义
# --------------------------------------------------------------------------- #
def test_none_scores_as_none_not_zero():
    """**未实测必须是 `None`，不是 0 分。** 这是本工程的立身之本。"""
    for m in ALL_METRICS:
        assert m.score(None) is None, f"{m.key} 把「未实测」当成了分数"
        assert m.score(float("nan")) is None, f"{m.key} 把 NaN 当成了分数"


def test_score_is_monotone_in_the_declared_direction():
    for m in ALL_METRICS:
        lo, hi = m.anchors[0][0], m.anchors[-1][0]
        s_lo, s_hi = m.score(lo), m.score(hi)
        if m.direction is Direction.LOWER_BETTER:
            assert s_lo >= s_hi, f"{m.key} 声明 lower-better 但小值分更低"
        elif m.direction is Direction.HIGHER_BETTER:
            assert s_lo <= s_hi, f"{m.key} 声明 higher-better 但大值分更低"


def _raw(**kw) -> RawResult:
    r = RawResult()
    r.values.update(kw)
    return r


def test_missing_metrics_do_not_enter_the_total():
    """只测一半维度时，总分**只由测到的那些**加权，而不是把未测的当 0 拖下去。"""
    r = _raw(**{"static.err_max_mrad": 0.0, "static.repeat_ptp_mrad": 0.1})
    c = score(r)
    assert c.total is not None
    # 静态维度的两条都接近满分 ⇒ 总分也必须接近满分
    assert c.total > 90, f"未测维度把总分拖到了 {c.total}"


def test_simulated_run_produces_no_total():
    """**仿真/桩**绝不出总分：桩不建模运动 ⇒ 退化读数会伪装成满分。"""
    r = _raw(**{"static.err_max_mrad": 0.0, "static.repeat_ptp_mrad": 0.1,
                "dyn.overshoot_mrad": 0.0, "torque.peak_ratio": 0.0})
    c = score(r, simulated=True)
    assert c.total is None, "仿真竟然出了总分 —— 这正是要堵死的假绿"
    assert c.scored_dimensions == []
    from litearm_benchmark.report import render_markdown
    md = render_markdown(c, r)
    assert "不出总分" in md, "报告顶部没有仿真横幅"


def test_trust_gate_excludes_whole_group_and_has_discriminating_power():
    """自证量不达标 ⇒ `track.*` **整组剔除**（不是给低分）。

    负控：把自证量抬到门槛之上，同一批数必须**重新计分**。
    """
    vals = {"track.rms_mrad": 1.0, "track.peak_mrad": 2.0, "track.lag_ms": 5.0}
    bad = _raw(**vals, **{"track.align_ratio": 1.05})       # 像噪声 ⇒ 不可信
    c = score(bad)
    unt = [m for d in c.dimensions for m in d.metrics if not m.trusted]
    assert any(m.spec.key == "track.rms_mrad" for m in unt), "自证量 1.05× 竟然不算不可信"
    assert c.metric("track.rms_mrad").score is not None      # 分数算出来了但被剔除
    assert not c.metric("track.rms_mrad").counted

    good = _raw(**vals, **{"track.align_ratio": 4.0})        # 真结构 ⇒ 可信
    c2 = score(good)
    assert c2.metric("track.rms_mrad").trusted, "自证量 4× 被误判为不可信"
    assert c2.metric("track.rms_mrad").counted


def test_excitation_gate_marks_motion_metrics_untested():
    """激励不足（臂基本没动）⇒ 运动类指标一律**未实测**，而不是给满分的退化读数。"""
    from litearm_benchmark.metrics.analysis import MIN_EXCITATION_RAD, RawResult
    r = RawResult()
    r.excitation_rad = MIN_EXCITATION_RAD / 10
    assert not r.excited
    r.excitation_rad = MIN_EXCITATION_RAD * 10
    assert r.excited
