"""笛卡尔动作下发的参数 —— **契约级**的检查，不需要真机。

2026-09-20 第一次上真机跑 `standard` 之前发现：`_cart` 给 `move_c` 的 `pose_start`
传的是**圆心**，而 core/firmware 的契约是**三点定圆**（起点 + via + 终点），
且 `pose_start` 会与**当前实测 TCP** 比对（容差 6mm / 0.03rad）。
圆心离实测 TCP 恒差一个半径（40mm）⇒ 真机上必抛 `InvalidCommandError`。

⚠ 这条错误的形状很典型：**参数个数对、类型对、名字也对，只有"语义"错**。
所以判据必须是"下发出去的那三个点**在不在同一个圆上、第一个点是不是实测 TCP**"，
而不是"函数被调用了"。
"""

import numpy as np
import pytest
from types import SimpleNamespace

from litearm_benchmark.suites.base import Action, Executor


class _Win:
    def __init__(self, n, label):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Cap:
    def window(self, n, label=""):
        return _Win(n, label)


class _Tcp:
    def __init__(self, p):
        self.value = list(p)


class _Arm:
    """只记下被问过的位姿；姿态恒为 0，位置由构造给定。"""

    def __init__(self, p0):
        self.p0 = list(p0)
        self.calls = []
        self.move_timeout = 0.05

    def get_tcp(self):
        return _Tcp(self.p0 + [0.0, 0.0, 0.0])

    def move_l(self, pose, speed=1.0, wait=True):
        self.calls.append(("move_l", [list(pose)]))
        return SimpleNamespace(n_wp=10, plan_us=1000)

    def move_c(self, s, v, g, speed=1.0, wait=True):
        self.calls.append(("move_c", [list(s), list(v), list(g)]))
        return SimpleNamespace(n_wp=20, plan_us=2000)

    def poll_cart(self):
        return SimpleNamespace(settled=True)


def _exec(p0):
    ex = Executor.__new__(Executor)
    ex.arm = _Arm(p0)
    ex.cap = _Cap()
    ex.trace = SimpleNamespace(tcp_t=np.zeros(0), tcp_p=np.zeros((0, 3)))
    ex.segments = []
    ex._log = lambda m: None
    return ex


#: 一个**远离基座轴**的 TCP —— 否则圆弧的"水平方向 u"会病态。
P0 = [0.30, 0.10, 0.40]


def test_cart_arc_starts_from_the_measured_tcp_not_from_the_center():
    """**负控**：第一点必须是实测 TCP；传圆心会让固件判"起点与实测 TCP 不符"而拒。"""
    ex = _exec(P0)
    ex._cart(Action("cart_arc", "圆弧", n_ticks=100, rest_s=0.0))

    kind, pts = ex.arm.calls[0]
    assert kind == "move_c"
    start = np.asarray(pts[0][:3])
    assert np.allclose(start, P0, atol=1e-6), \
        f"起点给成了 {start.tolist()}，实测 TCP 是 {P0}（差 {np.linalg.norm(start-np.asarray(P0))*1000:.1f} mm）"


def test_cart_arc_three_points_lie_on_one_circle_of_the_declared_radius():
    """三点必须**共圆**且半径 = 40mm —— 近共线三点会让固件扫出巨弧（本项目记录过）。"""
    ex = _exec(P0)
    ex._cart(Action("cart_arc", "圆弧", n_ticks=100, rest_s=0.0))
    _, pts = ex.arm.calls[0]
    a, b, c = (np.asarray(p[:3]) for p in pts)

    # 三点定圆：外接圆半径
    ab, cb = b - a, c - b
    n = np.cross(ab, cb)
    nn = float(np.linalg.norm(n))
    assert nn > 1e-9, "三点共线 ⇒ 定不出圆（固件会扫出巨弧）"
    r = (np.linalg.norm(ab) * np.linalg.norm(cb) * np.linalg.norm(c - a)) / (2 * nn)
    assert r == pytest.approx(0.04, rel=0.02), f"外接圆半径 {r*1000:.1f} mm ≠ 40 mm"

    # 起点必须是**圆上**的点，不是圆心本身 —— 外接圆半径会对，但圆心到起点的距离 =
    # 0 而不是半径。上一条测试查的是"起点 = 实测 TCP"，这条查"三点确实共圆"。
    assert np.linalg.norm(a - P0) < 1e-9, "起点不是实测 TCP"


def test_cart_line_sends_the_goal_only():
    """`move_l` 的起点由**固件**取实测 TCP，所以只该传**一个**目标位姿。"""
    ex = _exec(P0)
    ex._cart(Action("cart_line", "直线", n_ticks=100, rest_s=0.0, extra={"dy": 0.05}))
    kind, pts = ex.arm.calls[0]
    assert kind == "move_l" and len(pts) == 1
    goal = np.asarray(pts[0][:3])
    assert goal == pytest.approx(np.asarray(P0) + [0.0, 0.05, 0.0])


def test_cart_plan_failure_is_loud_but_does_not_kill_the_run():
    """规划失败是**测量结果**：记进段里并跳过，而不是让整档挂掉。"""
    ex = _exec(P0)

    def boom(*a, **k):
        raise RuntimeError("IK 不可达")

    ex.arm.move_l = boom
    ex._cart(Action("cart_line", "直线", n_ticks=100, rest_s=0.0))

    assert ex.segments and "IK 不可达" in ex.segments[-1].extra.get("error", ""), \
        "失败没有被记录 ⇒ 报告里会静默少一项（那正是「未实测要看得见」的反面）"


# --------------------------------------------------------------------------- #
# 5. 笛卡尔精度必须**逐段**算
# --------------------------------------------------------------------------- #
def _two_cart_segments():
    """一段真直线 + 一段真圆弧，各自的采样都精确落在**自己的**理论路径上。"""
    t, p, segs = [], [], []

    # A：直线 (0.30, 0.10, 0.40) → (0.30, 0.15, 0.40)
    a0, a1 = np.array([0.30, 0.10, 0.40]), np.array([0.30, 0.15, 0.40])
    for k in range(101):
        t.append(k * 0.01); p.append(a0 + (a1 - a0) * (k / 100.0))
    segs.append(SimpleNamespace(kind="cart_line", label="line", t0=0.0, t1=1.0,
                                extra={"poses": [list(a1) + [0.0, 0.0, 0.0]],
                                       "plan_us": 4800.0}))

    # B：z=0.5 平面内、圆心 (0,0,0.5)、半径 40mm 的 90° 弧
    c, r = np.array([0.0, 0.0, 0.5]), 0.04
    for k in range(101):
        th = (np.pi / 2) * k / 100.0
        t.append(1.5 + k * 0.01)
        p.append(c + r * np.array([np.cos(th), np.sin(th), 0.0]))
    ang = lambda d: list(c + r * np.array([np.cos(d), np.sin(d), 0.0])) + [0.0] * 3
    segs.append(SimpleNamespace(kind="cart_arc", label="arc", t0=1.5, t1=2.5,
                                extra={"poses": [ang(0.0), ang(np.pi / 4), ang(np.pi / 2)],
                                       "plan_us": 5600.0}))

    trace = SimpleNamespace(tcp_t=np.asarray(t), tcp_p=np.asarray(p))
    return trace, segs


def test_cart_deviation_is_per_segment_and_does_not_mix_actions():
    """**负控**：两段各自的采样都在自己的路径上 ⇒ 偏差都应 ≈0。

    旧实现把两段采样拼成一条线去拟合 ⇒ 直线段偏差会变成 **20 量级 mm**（真机实测 20.6）。
    """
    from litearm_benchmark.metrics.analysis import RawResult, _cartesian

    trace, segs = _two_cart_segments()
    r = RawResult()
    _cartesian(r, trace, segs)

    assert r.values["cart.line_dev_mm"] == pytest.approx(0.0, abs=1e-6), \
        "直线段偏差不为 0 ⇒ 采样被别的段污染了"
    assert r.values["cart.arc_dev_mm"] == pytest.approx(0.0, abs=1e-6), \
        "圆弧段径向偏差不为 0"


def test_the_naive_concatenation_would_have_been_wrong():
    """把两段拼起来算，误差必须**大得多** —— 这条证明上面那条判据有判别力。"""
    from litearm_benchmark.metrics.analysis import _dev_line

    trace, segs = _two_cart_segments()
    allp = trace.tcp_p
    # 旧算法：用**全部**采样、对**一条**端点连线
    naive = _dev_line(allp, allp[0], np.asarray(segs[0].extra["poses"][-1][:3]))
    assert naive > 0.010, f"拼接后的偏差只有 {naive*1000:.1f} mm —— 负控不成立"


def test_cart_plan_us_reports_the_worst_segment():
    from litearm_benchmark.metrics.analysis import RawResult, _cartesian
    trace, segs = _two_cart_segments()
    r = RawResult()
    _cartesian(r, trace, segs)
    assert r.values["cart.plan_us"] == pytest.approx(5600.0), "取最大值（最保守）"
    assert "最大值" in r.evidence["cart.plan_us"]


# --------------------------------------------------------------------------- #
# 6. 笛卡尔"停稳"判据 —— 两个条件缺一不可
# --------------------------------------------------------------------------- #
def _exec_with_tcp():
    return Executor.__new__(Executor)


def test_tcp_settle_requires_both_near_and_still():
    """**负控**：只看"没在动"会把启动前那一刻判成停稳（那时臂还没开始走）。"""
    ex = _exec_with_tcp()
    goal = np.array([0.0, 0.0, 0.5])
    tt = [i * 0.01 for i in range(100)]

    far_still = [list(goal + [0.05, 0.0, 0.0])] * 100        # 不动，但离目标 50mm
    assert not ex._tcp_settled(tt, far_still, goal), "离目标很远也算停稳？"

    near_moving = [list(goal + [0.05 * (1 - i / 99.0), 0.0, 0.0]) for i in range(100)]
    assert not ex._tcp_settled(tt, near_moving, goal), "还在动也算停稳？"

    near_still = [list(goal)] * 100
    assert ex._tcp_settled(tt, near_still, goal), "已到位且不动，应当判停稳"
