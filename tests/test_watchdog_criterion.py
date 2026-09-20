"""看门狗时延的**判据有没有判别力**。

2026-09-20 真机每档跑完都报「看门狗时延: 未观察到跳闸」，而那读起来像"通过"。
查固件后：**这个测法在常规状态下原理上观测不到它** ——

* `control_loop.c:717`：「`watchdog_kick()` **只在 `mode==MOVE_J` 时自动调用**」；
* `watchdog.c` 的 `[B]` 注释：「move_j 受控静止保持**持续 kick**」。

⇒ `movej` 到位后固件停在 MOVE_J 持位、每拍自 kick，"停止发命令"永远不会触发它。
我先前怀疑是自己的轮询在 kick —— **那是错的**：读查询不是运动命令，
`watchdog_kick()` 的调用点全在 `control_loop.c` 的运动命令里。

⛔ 且**不能去诱发它**：`control_loop.c:1787` 写明触发后是
「fail-soft 持位（**降刚度、τ=0、允许下垂**）」= **掉力矩**，
而本机 J4 距锁存线只有约 0.06 rad，一次下垂就可能越过而永久锁死。

⇒ 所以判据必须是：**MOVE_J 下如实报未实测 + 写明理由**，
而不是报一个读起来像"通过"的「未观察到」。
"""

from types import SimpleNamespace

from litearm_benchmark.metrics.analysis import RawResult, _safety
from litearm_benchmark.suites.base import Executor


def _exec(get_state, mode_holder=None):
    ex = Executor.__new__(Executor)
    ex.arm = SimpleNamespace(get_state=get_state)
    ex.trace = SimpleNamespace(meta={})
    ex.segments = []
    ex._log = lambda m: None
    ex._hold = lambda s: None
    return ex


def _state(mode, flags=0):
    return SimpleNamespace(value=SimpleNamespace(mode_name=mode, flags=flags))


# --------------------------------------------------------------------------- #
def test_move_j_reports_untested_with_a_reason_and_does_not_probe_for_a_trip():
    """**负控**：MOVE_J 下若报出一个数字（或什么都不说就返回），本判据就是失灵的。"""
    polls = []

    def get_state(refresh=False):
        polls.append(1)
        return _state("MOVE_J")

    ex = _exec(get_state)
    ex._watchdog(SimpleNamespace(label="看门狗时延", n_ticks=0))

    assert ex.trace.meta["watchdog_latency_s"] is None, \
        "MOVE_J 下不该给出时延（那是编的）"
    note = ex.trace.meta["watchdog_note"]
    assert "MOVE_J" in note and "自 kick" in note, f"理由没写清：{note}"
    assert len(polls) == 1, \
        f"只该读一次判模式；却读了 {len(polls)} 次 —— 说明还在傻等一个原理上不会来的事件"


def test_metric_carries_the_reason_into_the_skip():
    """未实测的理由必须**进到指标里** —— 否则报告上只看到"未实测"，看不到"为什么"。"""
    r = RawResult()
    _safety(r, SimpleNamespace(meta={
        "watchdog_latency_s": None,
        "watchdog_note": "MOVE_J 持位期间固件每拍自 kick ⇒ 观测不到",
    }), identity={})
    assert r.values.get("safe.watchdog_latency_s") is None
    assert "自 kick" in r.notes["safe.watchdog_latency_s"]


def test_non_move_j_mode_still_uses_the_passive_observation():
    """**正控**：换到别的模式（固件不自 kick）时，被动观察仍然有效、能抓到跳闸。"""
    n = {"i": 0}

    def get_state(refresh=False):
        n["i"] += 1
        return _state("MOVE_JS", flags=(0x2 if n["i"] > 3 else 0))

    ex = _exec(get_state)
    ex._watchdog(SimpleNamespace(label="看门狗时延", n_ticks=0))

    v = ex.trace.meta["watchdog_latency_s"]
    assert v is not None and v > 0, "非 MOVE_J 时应当能观察到跳闸"
    assert "watchdog_note" not in ex.trace.meta, "抓到跳闸就不该带「没抓到」的理由"


def test_safety_margin_evidence_names_the_latch_line_not_the_soft_limit():
    """`safe.margin_rad` 的出处必须写明是**距锁存线** —— 旧文案写的是「距软限」，是错的。"""
    r = RawResult()
    _safety(r, SimpleNamespace(meta={"min_margin_rad": 0.0597}), identity={})
    ev = r.evidence["safe.margin_rad"]
    assert "锁存线" in ev and "safety_check.c:165" in ev, ev
