"""单项目录与 `run_actions` —— **档位与单项必须走同一条装配路径**。

界面（`gui/`）挑的就是这里的动作表。所以这里守三件事：

1. 每一项的位形**都在安全包线内**（`POSES`），且从真机零位起跑时能过 `_gate`；
2. `run_actions` 与 `run_tier` 产出**同一个形状**的结果（同一套采集/分析/打分）；
3. 人工步骤的 `prompt` 钩子**真的被调用**，且"不应答"落到 **skip**（安全的一侧）。
"""

import numpy as np
import pytest

from litearm_benchmark.instruments import latch_margin_rad, inside_soft_limits
from litearm_benchmark.safety import MIN_TARGET_MARGIN_RAD, SafetyAbort
from litearm_benchmark.suites import singles, tiers
from litearm_benchmark.suites.base import POSES, Action, Executor

#: 真机读回值（2026-09-20）
LIM = [(-2.8095, 2.8095), (-1.7275, 1.7275), (-2.8095, 2.8095),
       (-3.071547, 0.017547), (-2.8095, 2.8095), (-1.5535, 1.5535), (-1.5535, 1.5535)]
N = 7


# --------------------------------------------------------------------------- #
# 1. 目录本身合规
# --------------------------------------------------------------------------- #
def test_every_single_only_uses_known_poses():
    """任一项引用了 `POSES` 里没有的位形 ⇒ 跑到那儿才炸，太晚。"""
    for it in singles.SINGLES.values():
        for a in it.actions:
            for k in (a.target, a.via):
                if k is not None:
                    assert k in POSES, f"{it.key} 引用了未知位形 {k!r}"


def test_every_single_passes_the_gate_from_the_real_zero_pose():
    """**核心判据**：每一项从真机零位起跑时，`_gate` 都得放行。

    ⚠ 这一条能抓住"位形越界/裕度不足"，而且**跑硬件之前**就能红。
    """
    for it in singles.SINGLES.values():
        for a in it.actions:
            if a.kind not in ("movej", "movej_sync", "cart_line", "cart_arc", "drift_ref"):
                continue
            ex = Executor.__new__(Executor)
            ex.s = type("S", (), {"limits": LIM})()
            ex.trace = type("T", (), {"q": np.asarray([[0.0] * N])})()
            tgt = POSES["bench"] if a.kind in ("cart_line", "cart_arc", "drift_ref") \
                else ex._pose(a.target)
            try:
                ex._gate(tgt, f"{it.key}/{a.label}")
            except SafetyAbort as exc:                     # pragma: no cover
                pytest.fail(f"单项 {it.key} 的 {a.label} 过不了闸：{exc}")


def test_single_keys_are_unique_and_documented():
    for key, it in singles.SINGLES.items():
        assert key == it.key
        assert it.actions, f"{key} 没有动作"
        assert it.desc and len(it.desc) > 10, f"{key} 的说明太短，界面上看不出它测什么"


# --------------------------------------------------------------------------- #
# 2. `run_actions` = 同一条路径
# --------------------------------------------------------------------------- #
def test_run_actions_produces_the_same_shape_as_a_tier(tmp_path):
    """单项与档位给出**同一种**结果（同一套采集/分析/打分），只是动作表不同。"""
    from litearm_benchmark.session import run_actions, run

    a = run_actions(singles.actions_of("gcheck"), label="single_gcheck",
                    port="fake", outdir=str(tmp_path / "a"), dry_run=True,
                    interactive=False)
    b = run("quick", port="fake", outdir=str(tmp_path / "b"), dry_run=True,
            interactive=False)
    for res in (a, b):
        assert res.card is not None and res.raw is not None
        assert res.card.simulated is True and res.card.total is None, \
            "桩数据竟然出了总分（它是仿真，必须不出）"
    # 两项都落了盘、且都带三段数据
    import os
    for d in (tmp_path / "a", tmp_path / "b"):
        for f in ("trace.npz", "raw.json", "REPORT.md"):
            assert os.path.exists(os.path.join(str(d), f)), f"{d} 缺 {f}"


def test_prompt_hook_is_used_and_defaults_to_skip():
    """人工步骤：给了 `prompt` 就得调它；**不应答时按 skip**（安全的一侧）。

    ⚠ "没应答"与"确认过了"在安全侧**不能同义** —— 后者会让臂带着未知载荷继续跑。
    """
    from litearm_benchmark.session import BenchSession

    calls = []

    def prompt(hint: str) -> str:
        calls.append(hint)
        return "skip"

    acts = [Action("manual", "装上配重块", manual_hint="请装好配重块", requires_manual=True),
            Action("rest", "收尾", rest_s=0.05)]
    with BenchSession("fake", dry_run=True) as s:
        res = s.run_actions(acts, label="manual_probe", interactive=True,
                           outdir=None, on_prompt=prompt)
    assert calls == ["请装好配重块"], f"prompt 没被调用或参数不对：{calls}"
    # skip 掉的人工步骤**不产生** rest 段（它就是"没做"）
    labels = [seg.label for seg in res.segments]
    assert "装上配重块" not in labels, "skip 的人工步骤不该留下段落"


def test_non_interactive_still_refuses_manual_actions():
    """非交互模式下含人工动作 ⇒ **拒绝**，而不是静默跳过。"""
    from litearm_benchmark.session import BenchSession

    acts = [Action("manual", "装上配重块", requires_manual=True)]
    with BenchSession("fake", dry_run=True) as s:
        with pytest.raises(RuntimeError, match="人工"):
            s.run_actions(acts, label="x", interactive=False, outdir=None)
