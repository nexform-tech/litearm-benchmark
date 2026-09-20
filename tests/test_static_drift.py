"""`endur.static_drift_mrad` —— 一条**此前没有任何数据来源**的指标。

2026-09-21 复核发现：`trace.meta['static_drift_mrad']` **全仓没有写入点**，
只有 `_endurance` 在读它。而它的 skip 理由却写着「本档未做首尾同点位对照
（**需 manual/完整档**）」—— 实际上 `full` 和 `manual_load` 也都不写它。
⇒ 它不是"要跑某个档才有"，是**一条从来没有数据来源的指标**。

现在由 suite 的 `drift_ref` 动作在各档首尾各记一次**同一点位**的快照
（多帧均值，压掉编码器量化），分析层取 `refs[0]` 与 `refs[-1]` 相减。

⚠ 判据必须逐轴取 |Δ静差| 的最大值 —— **不能**拿"后一次的静差"当漂移：
第一次本身可能就偏着（那是静差不是漂移），两者相减才是漂移。
"""

import numpy as np
import pytest

from litearm_benchmark.metrics.analysis import RawResult, _endurance

N = 7


def _trace(refs, *, t0=0.0, t1=100.0):
    return type("T", (), {
        "meta": {"drift_refs": refs} if refs is not None else {},
        "seq": np.zeros(0), "t_status": np.zeros(0),
    })()


def _ref(label, t, q, target):
    return {"label": label, "t": t, "q": list(q), "target": list(target), "n": 10}


def test_drift_is_the_difference_of_two_static_errors():
    """**关键**：第一次自己偏 5 mrad 不算漂移；只有「后 − 前」才是。"""
    tgt = [0.0] * N
    q0 = [0.005] + [0.0] * (N - 1)        # 第一次：J1 静差 +5 mrad
    q1 = [0.005, 0.0, -0.003] + [0.0] * (N - 3)   # 第二次：J1 没变，J3 偏 −3 mrad
    r = RawResult()
    _endurance(r, _trace([_ref("首", 0.0, q0, tgt), _ref("尾", 100.0, q1, tgt)]))

    v = r.values.get("endur.static_drift_mrad")
    assert v is not None, f"应当出数：{r.notes}"
    # J1 的 Δ静差 = 0（5→5），J3 的 Δ = −3 mrad ⇒ 最大值 = 3
    assert v == pytest.approx(3.0, abs=0.01), f"得到 {v}"
    assert "J3" in r.evidence["endur.static_drift_mrad"]
    assert "100 s" in r.evidence["endur.static_drift_mrad"], "要写明首尾间隔"


def test_unchanged_static_error_means_zero_drift():
    """**负控**：两次静差一模一样 ⇒ 漂移必须正好 0（而不是把静差本身报成漂移）。"""
    tgt = [0.0] * N
    q = [0.005, -0.002] + [0.0] * (N - 2)
    r = RawResult()
    _endurance(r, _trace([_ref("首", 0.0, q, tgt), _ref("尾", 60.0, q, tgt)]))
    assert r.values["endur.static_drift_mrad"] == pytest.approx(0.0, abs=1e-9)


def test_only_one_snapshot_is_reported_as_insufficient_not_as_zero():
    """**负控**：只记到 1 次 ⇒ **判未实测**（不能当 0 —— 那是"没测"不是"没漂"）。"""
    r = RawResult()
    _endurance(r, _trace([_ref("首", 0.0, [0.0] * N, [0.0] * N)]))
    assert r.values.get("endur.static_drift_mrad") is None
    assert "1 次" in r.notes["endur.static_drift_mrad"]


def test_no_snapshots_says_the_action_did_not_run():
    r = RawResult()
    _endurance(r, _trace(None))
    assert r.values.get("endur.static_drift_mrad") is None
    assert "drift_ref" in r.notes["endur.static_drift_mrad"]
