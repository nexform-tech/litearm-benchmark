"""`torque.g_match_pct` 的判据：**只在"模型预测得出来"的轴上比**。

2026-09-20 真机 `standard` 档这条指标报 **35.45%**，而注册表的历史分布是
**3.5 / 4.4 / 6.3 / 10.9 / 25.2%**（中位 6.3%）—— 远超上界。查下去发现**不是回归**，
是判据把两种工况混在一起了：

零位实测（同一台臂、同一时刻）：

| 轴 | `τ` 实测 | `G(q)` 模型 | 旧判据算出的"偏差" |
|---|---:|---:|---:|
| J1 | +0.4396 | **+0.0000** | **100%** |
| J3 | +0.3624 | **-0.0002** | **100%** |
| J2 | -1.4945 | -0.5679 | 62% |
| J4 | -0.8684 | -0.6797 | 22% |

⚠ 零位时 J1/J3 与重力**同向**，G 本来就是 0；实测那 0.44/0.36 N·m 是**库仑摩擦
撑着静平衡**的力矩。旧判据只按 `|τ| > 0.3` 挑轴 ⇒ 把"G 主导"和"摩擦主导"两种
**完全不同的工况**混在一起取中位数 ⇒ 指标被整体抬上去。

⇒ 门必须是 `|G(q_i)| ≥ max(|fc0_i|,|fc1_i|)`（该轴库仑摩擦）**且** `|τ_i| > 0.3`。
"""

import numpy as np
import pytest

from litearm_benchmark.metrics.analysis import RawResult, _torque

N = 7
#: 各轴库仑摩擦（N·m）—— 真机读回量级
FC = [0.30, 0.35, 0.12, 0.12, 0.03, 0.03, 0.03]


def _identity():
    return {"fc0": FC, "fc1": [-v for v in FC], "tau_max": [78.0] * 4 + [10.0] * 3}


def _trace(samples):
    """`_torque` 也要读 trace 的其它字段 —— 给空的（本测试只测 g_match 那一支）。"""
    return type("T", (), {
        "meta": {"g_samples": samples},
        "tau": np.zeros((0, N)), "q": np.zeros((0, N)), "dq": np.zeros((0, N)),
        "t_status": np.zeros(0), "n_joints": N,
    })()


def _smp(tau, g):
    return {"pose": "p", "q": [0.0] * N,
            "tau": list(tau), "g": list(g)}


def test_friction_dominated_axes_are_excluded():
    """**负控**：G≈0 而 τ 由摩擦撑着的轴必须被剔除，否则偏差会被算成 100%。"""
    samples = [
        # J1/J3：G 约为 0（与重力同向）⇒ 摩擦主导 ⇒ 必须剔除
        _smp([0.4396, -1.4945, 0.3624, -0.8684, 0.0, 0.0, 0.0],
             [0.0000, -0.5679, -0.0002, -0.6797, 0.0, 0.0, 0.0]),
    ]
    r = RawResult()
    _torque(r, _trace(samples), [], _identity())

    v = r.values.get("torque.g_match_pct")
    assert v is not None, f"不该判未实测：{r.notes}"
    # J1/J3 被剔除后，只剩 J2/J4：|(−1.4945+0.5679)/−1.4945| = 62%，
    # |(−0.8684+0.6797)/−0.8684| = 21.7% ⇒ 中位 = (62.0+21.7)/2 ≈ 41.9%
    # ⚠ 但**关键判据是"100% 那两根被剔了"** —— 见下一条。
    assert v < 70.0, f"摩擦主导的轴混进来了：{v}"
    assert "剔除" in r.evidence["torque.g_match_pct"]


def test_without_the_gate_the_value_would_be_dragged_to_100():
    """同一份数据，**去掉门**就会得到 100% —— 这条证明上面那条判据有判别力。"""
    tau = [0.4396, -1.4945, 0.3624, -0.8684, 0.0, 0.0, 0.0]
    g = [0.0000, -0.5679, -0.0002, -0.6797, 0.0, 0.0, 0.0]
    naive = [abs(t - gg) / abs(t) for t, gg in zip(tau, g) if abs(t) > 0.3]
    assert naive and max(naive) > 0.99, "旧判据本该把 J1/J3 算成 ~100%"


def test_skips_when_no_axis_has_a_dominant_gravity_term():
    """**负控**：一个"重力项主导"的轴都没有 ⇒ 判**未实测**，而不是给个假数。"""
    samples = [_smp([0.40, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0],
                    [0.001, 0.0, -0.001, 0.0, 0.0, 0.0, 0.0])]
    r = RawResult()
    _torque(r, _trace(samples), [], _identity())

    assert r.values.get("torque.g_match_pct") is None
    assert "重力项主导" in r.notes["torque.g_match_pct"]


def test_skips_when_friction_is_unreadable():
    """读不到 `fc0/fc1` ⇒ **判不了**（而不是退回旧的无门算法）。"""
    r = RawResult()
    _torque(r, _trace([_smp([1.0] * N, [-1.0] * N)]), [], {"tau_max": [78.0] * 4 + [10.0] * 3})
    assert r.values.get("torque.g_match_pct") is None
    assert "fc0/fc1" in r.notes["torque.g_match_pct"]
