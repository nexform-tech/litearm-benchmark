"""信号原语与跨流对齐 —— 这两层最容易出"看不见的错"，故每条都配负控。"""
import numpy as np
import pytest

from litearm_benchmark.align import align, moving_axes
from litearm_benchmark.metrics import signals as S


# --------------------------------------------------------------------------- #
def test_quant_floor_and_variance_subtraction():
    """量化地板 = LSB/√12；扣地板必须**按方差相减**，不是直接减。"""
    assert S.quant_floor_std(0.0586) == pytest.approx(0.0586 / 12 ** 0.5)
    # 独立方差源：σ² = σ真² + σ地板²
    floor, true = 0.01, 0.03
    measured = (true ** 2 + floor ** 2) ** 0.5
    assert S.excess_over_floor(measured, floor) == pytest.approx(true, rel=1e-9)
    # 小于地板 ⇒ 夹到 0，不许出负值
    assert S.excess_over_floor(0.005, floor) == 0.0


def test_cruise_segment_takes_the_longest_run_not_first_to_last():
    """**负控**：末尾戳一个尖峰，巡航段**不许**被它拖长。

    用"首末超阈下标"的写法会把这个尖峰算进去 ⇒ 窗口跨进减速段 ⇒ 纹波虚高 12 倍
    （本项目真实事故）。
    """
    v = np.zeros(200)
    v[50:130] = 1.0            # 真正的巡航段
    v[195] = 1.5               # 收尾尖峰
    lo, hi = S.cruise_segment(v, 0.9)
    assert (lo, hi) == (50, 129), f"巡航段被尖峰污染：{(lo, hi)}"


def test_motion_window_is_not_fooled_by_the_hold_tail():
    """按**速度**界定运动段 —— 用"位置离开起点"会把收尾保持段并进来（真实事故）。"""
    t = np.arange(0, 4.0, 1 / 300)
    v = np.zeros_like(t)
    v[100:150] = 1.0
    lo, hi = S.motion_window(v)
    assert lo == 100 and hi == 149


def test_fit_line_residual_ignores_linear_drift():
    """静止抖动用**直线拟合残差**：缓慢漂移要被扣掉，只留真抖动。"""
    t = np.linspace(0, 2.0, 400)
    drift = 5e-3 * t                      # 5 mrad/s 的热漂
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 2e-4, t.size)
    std = S.fit_line_residual_std(t, drift + noise)
    assert std == pytest.approx(2e-4, rel=0.2), f"漂移没被扣掉：{std}"
    assert np.std(drift + noise) > 2e-3, "负控失效：不拟合时 std 本该很大"


def test_seq_reconcile_counts_lost_frames_and_reports_pair_count():
    """16-bit `seq` 会回绕；判据是「Δseq − 1」，**必须同报「可比对数」**（0 对不能判绿）。"""
    # 回绕：必须**逐帧连续**才叫"没丢"
    gap, pairs = S.seq_reconcile([65534, 65535, 0, 1])
    assert pairs == 3 and gap == 0.0, f"回绕没处理：gap={gap}"
    # 真丢帧：Δseq=2 ⇒ 丢 1 帧；Δseq=4 ⇒ 丢 3 帧
    assert S.seq_reconcile([10, 11, 13, 14])[0] == 1.0
    assert S.seq_reconcile([1, 5])[0] == 3.0
    # 帧太少 ⇒ 0 对（"没有可比对"与"没有缺口"不是同一句话）
    assert S.seq_reconcile([7])[1] == 0


def test_seq_reconcile_ignores_host_read_gaps():
    """**负控**：主机 0.3s 没读但 `Δseq=1` ⇒ 缺口必须判 **0**。

    固件的状态帧是**被动**的（主机不读它基本不发），所以 `dt` 里那段"没读"的时间
    **不该**算成"本该有帧"。旧判据（`dt × 100Hz`）在同一份数据上会报 ~30 帧 ——
    真机实测：`full` 档里 `seq` 差 24383 / 收到 24384 帧（**一帧没丢**），
    旧算法却报 **37.46 帧、11 分**（假红）；那几处的 `dt` 全是 ~300ms，
    正好是 `_at_rest()` 每 300ms 读一次的位置。
    """
    seq = [1, 2, 3]                       # 逐帧连续
    gap, pairs = S.seq_reconcile(seq)
    assert gap == 0.0 and pairs == 2

    # 旧算法（掺 dt）在同一份数据上给出的数：
    old = abs((seq[1] - seq[0]) % 65536 - 0.30 * 100.0)
    assert old > 25.0, f"旧算法本该报 ~{old:.1f} 帧 ⇒ 本判据的判别力成立"


def test_overshoot_and_rebound_are_different_quantities():
    """越峰与回摆是两个量；回摆**恒 ≥ 0**（曾把代数量当回摆，符号是反的）。"""
    t = np.linspace(0, 1, 200)
    q = 0.5 * (1 - np.cos(np.pi * np.clip(t / 0.5, 0, 1)))   # 走到 1.0
    q[100:130] += 0.02                                        # 冲过头
    peak, rebound = S.overshoot_rebound(q, 1.0, +1.0)
    assert peak > 0
    assert rebound >= 0


# --------------------------------------------------------------------------- #
def _jerk_limited(t, t0, dur, amp):
    """一段"像 movej"的 jerk 受限平滑阶跃 —— 真机信号的关键是**快变沿**。

    ⚠ 别用慢正弦测对齐：0.3~0.4Hz 的信号错开 80ms 几乎没差别 ⇒ 自证量区分不开，
    会误判成"对齐不可信"（我第一版就是这么写的，红的是测试不是实现）。
    """
    u = np.clip((t - t0) / dur, 0.0, 1.0)
    return amp * (u * u * (3 - 2 * u))


def test_align_recovers_known_delay_and_has_a_self_proving_ratio():
    """对齐要能**找回已知时延**，且自证量在真结构下明显大于 1。"""
    dt = 1 / 300
    t_ref = np.arange(0, 3.0, dt)
    rng = np.random.default_rng(1)
    q_ref = np.stack([
        _jerk_limited(t_ref, 0.30, 0.35, 0.60) + _jerk_limited(t_ref, 1.40, 0.35, -0.45),
        _jerk_limited(t_ref, 0.55, 0.35, -0.50) + _jerk_limited(t_ref, 2.00, 0.35, 0.35),
    ], axis=1)
    true_delay = 0.043
    t_m = np.arange(0, 3.0, 0.01)
    q_m = np.stack([np.interp(t_m - true_delay, t_ref, q_ref[:, i]) for i in range(2)],
                   axis=1) + rng.normal(0, 2e-4, (t_m.size, 2))
    res = align(t_m, q_m, t_ref, q_ref)
    assert res.ok, res.why_bad()
    # ⚠ **符号约定**：δ 是"把**实测**时间轴平移多少才能对上参考"。
    #   实测**滞后**参考 43ms ⇒ δ = **−0.043**（我第一版写成 +0.043，是把符号想反了）。
    #   真机观测与此一致：动作电池里 δ 落在 −8 ~ −30ms（伺服滞后于指令）。
    assert res.delta == pytest.approx(-true_delay, abs=0.01), f"δ={res.delta}"
    assert res.ratio > 2.0


def test_align_rejects_unrelated_streams():
    """**负控**：两条**无关**的流，自证量必须接近 1 ⇒ `ok` 为假。

    没有这条，"对齐成功"就没有意义 —— 它只是两个数碰巧都对上而已。
    """
    rng = np.random.default_rng(2)
    t_ref = np.arange(0, 3.0, 1 / 300)
    t_m = np.arange(0, 3.0, 0.01)
    q_ref = rng.normal(0, 0.3, (t_ref.size, 2))
    q_m = rng.normal(0, 0.3, (t_m.size, 2))
    res = align(t_m, q_m, t_ref, q_ref)
    assert not res.ok, f"两条无关的流竟然判成对齐成功（ratio={res.ratio:.2f}）"


def test_align_flags_delta_pinned_at_window_edge():
    """δ 顶在搜索窗边缘 ⇒ 必须报 `at_edge`（真偏移可能在窗外 —— 真实事故）。"""
    t_ref = np.arange(0, 3.0, 1 / 300)
    # ⚠ 信号必须**非周期**：我第一版用 0.5Hz 正弦平移 1.0s —— 那正好是两个周期，
    #   等于没平移，δ=0 反而是对的。
    q_ref = np.stack([_jerk_limited(t_ref, 0.40, 0.35, 0.6)], axis=1)
    t_m = np.arange(0, 3.0, 0.01)
    # 平移 1 秒（远超 ±150ms 的窗）⇒ 只能顶在窗边
    q_m = np.stack([np.interp(t_m + 1.0, t_ref, q_ref[:, 0])], axis=1)
    res = align(t_m, q_m, t_ref, q_ref)
    assert res.at_edge, f"δ={res.delta*1e3:.1f}ms 应当被判为顶窗"


def test_moving_axes_ignores_static_joints():
    t = np.arange(0, 1.0, 1 / 300)
    q = np.stack([np.sin(t) * 0.5, np.full_like(t, 0.3), t * 0.2], axis=1)
    ax = moving_axes(t, q)
    assert 0 in ax and 2 in ax and 1 not in ax
