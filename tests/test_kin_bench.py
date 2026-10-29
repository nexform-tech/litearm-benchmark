"""固件自检（`0x49`）的取数与两个量的分辨。

2026-09-20 真机跑 `standard` 时「固件计算性能」整维 **0/2 未实测**，日志只报出
`{'_frames': 1}`。两个独立的错：

1. `kin_bench()` 返回 `Msg` **信封**，我读的是信封本身而不是 `.value`
   ⇒ `timings` 恒为空（core 里 11 个读型入口都包了信封，本工程其它地方都写了
   `.value`，只有这处漏了）。
2. 拿 `timings["LOOP1"]` 当「控制拍峰值」。`LOOP1` 是自检里**禁中断单次**测的控制步
   计算量，而固件另有 `loop_k = g_arm.loop_max_cyc/1000`（"历史最长单拍耗时"）。
   真机实测 **20.06 vs 199 kcycle**，差 10×。

⚠ 以及一条来自项目历史的教训：`.get(k, 0)` 式出口会让**「读不到」与「真的是 0」**
长得一模一样 —— 所以「LINK 帧没到」时必须判**未实测**，不许填 0。
"""

import pytest
from types import SimpleNamespace

from litearm_benchmark.metrics.analysis import RawResult, _compute
from litearm_benchmark.suites.base import Executor

#: 真机实测值（2026-09-20）
LOOP1_CYC = 20060          # 禁中断单次测的控制步计算量（cycle）
LOOP_K = 199               # 真实整拍峰值（kcycle）


def _result(*, with_link: bool):
    link = {"loop_k": LOOP_K, "ovr": 0, "rxl0": 0, "rxl1": 0, "crc": 0} if with_link else {}
    return SimpleNamespace(
        timings={"LOOP1": (LOOP1_CYC,), "FK": (200, 47203, 47418)},
        link=link,
        loop_max_kcycle=LOOP_K, loop_overruns=0,
        crc_errors=0, reply_dropped=0, can_tx_fail=40,
    )


class _Diag:
    def __init__(self, res):
        self._res = res

    def kin_bench(self):
        return SimpleNamespace(value=self._res)          # ← 信封


def _exec(res):
    ex = Executor.__new__(Executor)
    ex.arm = SimpleNamespace(diag=_Diag(res))
    ex.trace = SimpleNamespace(meta={})
    ex._log = lambda m: None
    return ex


# --------------------------------------------------------------------------- #
def test_envelope_is_unwrapped_and_loop_k_wins_over_loop1():
    """**负控**：不拆信封 ⇒ 什么都读不到；读错量 ⇒ 会拿到 20 而不是 199。"""
    ex = _exec(_result(with_link=True))
    ex._kin_bench(SimpleNamespace(label="自检", n_ticks=200))

    kb = ex.trace.meta["kin_bench"]
    assert kb.get("loop_max_kcycle") == float(LOOP_K), \
        f"控制拍峰值取成了 {kb.get('loop_max_kcycle')} —— 应是 loop_k={LOOP_K}（不是 LOOP1/1000）"
    assert kb.get("loop1_isolated_kcycle") == pytest.approx(20.06), \
        "LOOP1 只该当旁证记下来"
    assert kb["_frames"] == 2


def test_link_frame_missing_never_fills_zero():
    """**负控**：只收到 1 帧时，控制拍峰值/越拍数/链路计数**一律不写**。

    `.get(k, 0)` 式出口会把「读不到」伪装成「很健康」—— 本项目栽过这个。
    """
    ex = _exec(_result(with_link=False))
    ex._kin_bench(SimpleNamespace(label="自检", n_ticks=200))

    kb = ex.trace.meta["kin_bench"]
    assert kb["_frames"] == 1
    assert "loop_max_kcycle" not in kb, "LINK 帧没到却给出了控制拍峰值"
    assert "overruns" not in kb, "LINK 帧没到却给出了越拍数"
    for meta in ("crc_errors", "reply_dropped", "can_tx_fail", "rx_fifo_lost"):
        assert meta not in ex.trace.meta, f"LINK 帧没到却写了 {meta}"


def test_compute_skips_instead_of_scoring_zero_when_link_missing():
    """**负控**：读不到 ⇒ `skip`（未实测），**不是** 0 分。"""
    r = RawResult()
    _compute(r, {"_frames": 1})                      # 只有 _frames，没有任何数
    assert r.values.get("compute.loop_max_kcycle") is None
    assert "未实测" in r.notes["compute.loop_max_kcycle"]
    assert r.values.get("compute.overruns") is None

    r2 = RawResult()
    _compute(r2, {"loop_max_kcycle": float(LOOP_K), "overruns": 0,
                  "loop1_isolated_kcycle": 20.06, "_frames": 2})
    assert r2.values["compute.loop_max_kcycle"] == float(LOOP_K)
    assert r2.values["compute.overruns"] == 0.0
    assert "旁证" in r2.evidence["compute.loop_max_kcycle"]
