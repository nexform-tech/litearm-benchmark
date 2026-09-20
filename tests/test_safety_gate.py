"""安全闸的判据 —— 裕度**量到哪条线**。

这一整个文件是为 2026-09-20 那次真机首跑写的：benchmark 第一次上真机就**一步也
跑不起来**，因为闸门把裕度量到了**软限**，而固件真正锁存的位置在 `软限 ± 0.05`。
J4 的软限上界只有 +0.0175、零位天然距它 0.0097 ⇒ 三道判据（0.05/0.02/0.01）
**全部恒不可达**。这是本项目记录过四次的同一类错（"判据低于物理地板 / 恒不可达"）。

⚠ 判据改动的验证要点：**改后的判据必须仍然在该红时红**。所以下面既有"J4 正常零位
必须放行"，也有"顶到锁存线上必须拒绝"。
"""

import numpy as np
import pytest
from types import SimpleNamespace

from litearm_benchmark import instruments as I
from litearm_benchmark.instruments import latch_margin_rad
from litearm_benchmark.safety import SafetyAbort, preflight

#: 真机读回值（2026-09-20，`0x24` 设备当天值；与固件 `joint_limit_macros.h:15-16` 一致）
J4_LO, J4_HI = -3.071547, +0.017547
#: 真机零位实测 `q4`
J4_AT_ZERO = +0.0078

#: 其余轴取一组宽松限位即可（它们不是本文件的被测对象）
LIM = [(-2.8095, 2.8095), (-1.7275, 1.7275), (-2.8095, 2.8095),
       (J4_LO, J4_HI), (-2.8095, 2.8095), (-1.5535, 1.5535), (-1.5535, 1.5535)]


# --------------------------------------------------------------------------- #
# 1. 裕度本身
# --------------------------------------------------------------------------- #
def test_latch_margin_is_measured_to_the_latch_line_not_the_soft_limit():
    """J4 正常零位：距**锁存线** 0.0597，距**软限**只有 0.0097。

    这两个数差正好一个 `POS_LATCH_BAND`(=0.05)。**旧实现量的是后者**，于是 0.0097
    连闸门要求的 0.05 都够不到 ⇒ 恒拒绝。
    """
    m_latch = latch_margin_rad(J4_AT_ZERO, J4_LO, J4_HI)
    m_soft = min(J4_AT_ZERO - J4_LO, J4_HI - J4_AT_ZERO)

    assert m_latch == pytest.approx(m_soft + I.POS_LATCH_BAND.value, abs=1e-6)
    assert m_soft < 0.02, "软限裕度本该很小 —— 这正是旧判据失败的原因"
    assert m_latch > 0, f"J4 零位距锁存线只有 {m_latch:.4f} ⇒ 恒拒绝，benchmark 跑不起来"


def test_latch_margin_zero_on_the_line_and_negative_beyond():
    """在线上 = 0，线外 < 0。**符号必须能把"还能动"与"要锁存了"分开。**"""
    assert latch_margin_rad(J4_HI + I.POS_LATCH_BAND.value, J4_LO, J4_HI) == pytest.approx(0.0)
    assert latch_margin_rad(J4_HI + 0.10, J4_LO, J4_HI) < 0
    assert latch_margin_rad(J4_LO - 0.10, J4_LO, J4_HI) < 0


# --------------------------------------------------------------------------- #
# 2. 闸门行为 —— 放行与拒绝各一条
# --------------------------------------------------------------------------- #
def _arm(q4: float, *, enabled: bool):
    st = SimpleNamespace(
        joints=[SimpleNamespace(q=0.0) for _ in range(7)],
        joint_fault=0, flags=0, enabled=enabled, drop_hold_inferred=False,
    )
    st.joints[3].q = q4
    return SimpleNamespace(get_state=lambda refresh=True: SimpleNamespace(value=st))


def test_preflight_passes_on_the_real_zero_pose():
    """**真机零位必须放行** —— 这一条就是那个"跑不起来"的回归判据。"""
    rep = preflight(_arm(J4_AT_ZERO, enabled=False), limits=LIM)
    assert rep.ok, f"真机零位被拒：{rep.faults}（{rep.render()}）"
    assert rep.tightest == 4, "最紧的轴应当是 J4"
    assert not rep.faults


def test_preflight_still_refuses_a_joint_on_the_latch_line():
    """**负控**：把 J4 挪到锁存线上，同一段代码必须拒绝。否则上面那条绿毫无意义。"""
    on_line = J4_HI + I.POS_LATCH_BAND.value
    rep = preflight(_arm(on_line, enabled=True), limits=LIM)
    assert not rep.ok
    assert any("锁存线" in f for f in rep.faults), f"没报锁存线：{rep.faults}"


def test_outside_soft_limits_but_inside_the_latch_line_never_latches():
    """软限外、但**未越锁存线** ⇒ 固件**根本不进锁存分支**，**两态都不锁存**。

    这是"失能的臂漂出软限"这个**常规**状态的正确处置：告警、放行，
    而不是拒绝 —— 拒绝会把一个每天都会发生的可恢复态变成"永远跑不了"。
    """
    outside = J4_HI + 0.02          # 软限外 0.02，锁存线在 +0.0675 ⇒ 距线 0.03 > 0

    for enabled in (False, True):
        rep = preflight(_arm(outside, enabled=enabled), limits=LIM)
        assert rep.ok, f"enabled={enabled} 时被误拒：{rep.faults}"
        assert any("软限外" in w for w in rep.warnings), \
            f"enabled={enabled} 时软限外没告警：{rep.warnings}"


def test_beyond_the_latch_line_splits_on_enabled_and_on_the_recover_band():
    """越过锁存线之后才分两态 —— 判据逐字抄固件 `safety_check.c:183`：

    * **失能** 且越线幅度 ≤ `POS_RECOVER_BAND`(0.05) ⇒ 不锁存（`enable` 受控回程）⇒ 放行；
    * 其余（已使能，或越线幅度 > 0.05）⇒ 5 拍后锁存 ⇒ 拒绝。
    """
    just_over = J4_HI + I.POS_LATCH_BAND.value + 0.02      # 越线 0.02
    way_over = J4_HI + I.POS_LATCH_BAND.value + 0.20       # 越线 0.20

    limp_ok = preflight(_arm(just_over, enabled=False), limits=LIM)
    assert limp_ok.ok, f"失能 + 越线 0.02 应当放行（固件会回程）：{limp_ok.faults}"
    assert any("锁存线" in w for w in limp_ok.warnings), \
        f"越线必须告警：{limp_ok.warnings}"

    # **负控 ①**：同一位置，改成已使能 ⇒ 必须拒绝
    assert not preflight(_arm(just_over, enabled=True), limits=LIM).ok, \
        "已使能时越线竟然放行 —— 固件对使能态不适用回程宽限"

    # **负控 ②**：失能，但越线幅度超过回程带 ⇒ 必须拒绝
    assert not preflight(_arm(way_over, enabled=False), limits=LIM).ok, \
        "失能但越线 0.20 > POS_RECOVER_BAND 竟然放行"


# --------------------------------------------------------------------------- #
# 3. 动作闸（`Executor._gate`）—— 位形表必须真的过得了
# --------------------------------------------------------------------------- #
def _gate(target):
    from litearm_benchmark.suites.base import Executor
    ex = Executor.__new__(Executor)
    ex.s = SimpleNamespace(limits=LIM)
    ex.trace = SimpleNamespace(q=np.asarray([J4_AT_ZERO] + [0.0] * 6, float)[None, :])
    ex._gate(target, "test")


def test_every_benchmark_pose_passes_the_gate_from_the_real_zero_pose():
    """位形表**全部**动作都得能从真机零位起跑 —— 这是"能不能跑"的直接判据。"""
    from litearm_benchmark.suites.base import POSES
    for key, pose in POSES.items():
        try:
            _gate(pose)
        except SafetyAbort as exc:                                # pragma: no cover
            pytest.fail(f"位形 {key} 从真机零位被拒：{exc}")


def test_gate_refuses_a_target_outside_the_soft_limits():
    """**负控**：目标在软限外必须拒 —— 固件会把 `q_ref` 钳回，那一动作不会真的到位。"""
    from litearm_benchmark.suites.base import POSES
    bad = list(POSES["zero"])
    bad[3] = J4_HI + 0.5                       # J4 越过软限 0.5
    with pytest.raises(SafetyAbort, match="软限外"):
        _gate(bad)
