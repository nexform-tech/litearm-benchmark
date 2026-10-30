"""Offline tests for litearm_benchmark.torque_eval — no hardware, no numpy/matplotlib."""
from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from litearm_benchmark.torque_eval import cli, filters, rnea
from litearm_benchmark.torque_eval.analyze import AnalyzeOptions, analyze_session
from litearm_benchmark.torque_eval.cli import _write_overview
from litearm_benchmark.torque_eval.collect import (
    Sample, ScenarioSpec, Session, run_scenario, validate_scenario)
from litearm_benchmark.torque_eval.model import factory_snapshot
from litearm_benchmark.torque_eval.plots import generate_all
from litearm_benchmark.torque_eval.report import write_report


# ---------------------------------------------------------------------------
# RNEA: firmware golden probe (dyn.c MODEL_PROBE_Q / G_FACTORY_PROBE)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q, expected_max", [
    ([0.0, -0.22, 0.94, 0.0, 0.0, 0.0, 0.0], 2.1340647),
    ([0.0, 0.62, -1.60, 0.0, 0.0, 0.0, 0.0], 7.5729609),
])
def test_rnea_matches_firmware_gravity_probe(q, expected_max):
    snap = factory_snapshot()
    g = rnea.gravity_torque(q, snap.effective_bodies(), (0.0, 0.0, -9.81))
    assert max(abs(v) for v in g) == pytest.approx(expected_max, abs=1.5e-4)


def _com_positions(q, bodies):
    """独立前向运动学: 返回 9 个刚体质心在世界系的位置。"""
    Rw = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]] + [None] * 8
    p = [[0.0, 0.0, 0.0] for _ in range(9)]
    out = []
    for i in range(1, 9):
        pa = rnea.DYN_PARENT[i]
        rwpf = rnea._matmul(Rw[pa], rnea._FIX_IO[i])
        sgn = 1.0 if rnea.DYN_AXIS[i][2] >= 0.0 else -1.0
        Rw[i] = (rnea._matmul(rwpf, rnea._zrot(sgn * float(q[i - 1])))
                 if rnea.DYN_JTYPE[i] == 0 else rwpf)
        tmp = rnea._matvec(Rw[pa], rnea.DYN_FIX_XYZ[i])
        p[i] = [p[pa][k] + tmp[k] for k in range(3)]
        com = rnea._matvec(Rw[i], bodies[i].com)
        out.append([p[i][k] + com[k] for k in range(3)])
    return out


def _gravity_by_potential(q, bodies, g):
    """G = ∂U/∂q, U = −Σ m·g·p_com; 用势能数值梯度独立校验符号与幅值。"""
    def potential(qq):
        return -sum(b.mass * sum(g[k] * c[k] for k in range(3))
                    for b, c in zip(bodies[1:], _com_positions(qq, bodies)))
    h = 1e-6
    out = []
    for j in range(7):
        qp, qm = list(q), list(q)
        qp[j] += h
        qm[j] -= h
        out.append((potential(qp) - potential(qm)) / (2.0 * h))
    return out


@pytest.mark.parametrize("q", [
    [0.0, -0.22, 0.94, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.62, -1.60, 0.0, 0.0, 0.0, 0.0],
    [0.1, -0.2, 0.3, -0.4, 0.2, 0.1, -0.1],
])
def test_gravity_matches_potential_energy_gradient(q):
    snap = factory_snapshot()
    bodies = snap.effective_bodies()
    g = (0.0, 0.0, -9.81)
    got = rnea.gravity_torque(q, bodies, g)
    want = _gravity_by_potential(q, bodies, g)
    for a, b in zip(got, want):
        assert a == pytest.approx(b, rel=1e-5, abs=1e-6)


def test_mass_matrix_is_consistent_with_rnea():
    snap = factory_snapshot()
    bodies = snap.effective_bodies()
    q = [0.1, -0.2, 0.3, -0.4, 0.2, 0.1, -0.1]
    ddq = [0.5, -0.3, 0.2, 0.1, -0.2, 0.3, -0.1]
    M = rnea.mass_matrix(q, bodies)
    tau = rnea.inertia_torque(q, ddq, bodies)
    pred = [sum(M[i][j] * ddq[j] for j in range(7)) for i in range(7)]
    for a, b in zip(tau, pred):
        assert a == pytest.approx(b, abs=1e-9)


def test_coriolis_identity_holds():
    """dqᵀ(Ṁ − 2C)dq ≈ 0 —— 独立于重力项, 能抓住 C 恒为 0 或符号/系数错误。"""
    snap = factory_snapshot()
    bodies = snap.effective_bodies()
    q = [0.3, -0.4, 0.5, -0.2, 0.1, -0.3, 0.2]
    dq = [0.4, -0.3, 0.2, 0.5, -0.1, 0.2, -0.25]

    # coriolis_torque 返回 C(q,dq)·dq; 二次型 dqᵀC dq 就是 dq·(C dq)。
    Cdq = rnea.coriolis_torque(q, dq, bodies)
    h = 1e-6
    qp = [q[i] + h * dq[i] for i in range(7)]
    qm = [q[i] - h * dq[i] for i in range(7)]
    Mp = rnea.mass_matrix(qp, bodies)
    Mm = rnea.mass_matrix(qm, bodies)
    Mdot = [[(Mp[i][j] - Mm[i][j]) / (2.0 * h) for j in range(7)]
            for i in range(7)]
    dM = sum(dq[i] * sum(Mdot[i][j] * dq[j] for j in range(7))
             for i in range(7))
    dC = sum(dq[i] * Cdq[i] for i in range(7))
    assert abs(dM - 2.0 * dC) < 1e-5 * max(1.0, abs(dM))


def test_theory_torques_total_is_sum_of_parts():
    snap = factory_snapshot()
    q = [0.05, -0.1, 0.2, -0.15, 0.1, 0.05, -0.05]
    dq = [0.1, -0.2, 0.15, 0.05, -0.1, 0.2, -0.05]
    ddq = [0.3, -0.1, 0.2, 0.05, -0.15, 0.1, -0.2]
    d = rnea.theory_torques(snap, q, dq, ddq)
    for i in range(7):
        assert d["total"][i] == pytest.approx(
            d["gravity"][i] + d["coriolis"][i] + d["inertia"][i]
            + d["jm"][i] + d["friction"][i], abs=1e-12)


# ---------------------------------------------------------------------------
# Filters / resampling / metrics
# ---------------------------------------------------------------------------

def test_uniform_grid_and_linear_resample():
    grid = filters.uniform_grid(0.0, 1.0, 10.0)
    assert len(grid) == 11
    assert grid[-1] == pytest.approx(1.0)
    out = filters.linear_resample([0.0, 1.0], [0.0, 2.0], [0.0, 0.5, 1.0])
    assert out == [0.0, 1.0, 2.0]
    # 端点外钳住
    assert filters.linear_resample([0.0, 1.0], [3.0, 4.0], [-1.0, 2.0]) == [3.0, 4.0]


def test_zero_phase_lowpass_preserves_dc_and_phase():
    fs = 100.0
    n = 1000
    x = [1.0] * n
    y = filters.lowpass_zero_phase(x, fs, 5.0)
    # 稳态直流增益必须为 1 (边缘反射填充的过渡段会有 ~1e-6 残差, 跳过)
    assert min(y[100:-100]) == pytest.approx(1.0, abs=1e-12)

    t = [k / fs for k in range(n)]
    s = [math.sin(2 * math.pi * 1.0 * ti) for ti in t]
    y = filters.lowpass_zero_phase(s, fs, 10.0)
    assert filters.pearson(s[50:-50], y[50:-50]) > 0.9999
    # 零相位: 峰值时刻不移动 (普通因果滤波会右移)
    # 1 Hz 信号每 100 点一个峰, 取窗口中段的单个峰以避免周期重复取到别的波峰
    lo, hi = 400, 450
    i_s = max(range(lo, hi), key=lambda i: s[i])
    i_y = max(range(lo, hi), key=lambda i: y[i])
    assert abs(i_y - i_s) <= 1


def test_lowpass_short_series_preserves_index():
    """pad 被截断到 n-1 时切片也必须用同一 pad, 否则输出整体偏移。"""
    n = 30
    x = [0.0] * n
    x[15] = 1.0
    y = filters.lowpass_zero_phase(x, 100.0, 5.0)
    assert len(y) == n
    assert max(range(n), key=lambda i: y[i]) == 15


def test_central_diff_matches_analytic_derivative():
    fs = 200.0
    f = 2.0
    t = [k / fs for k in range(400)]
    x = [math.sin(2 * math.pi * f * ti) for ti in t]
    dx = filters.central_diff(x, 1.0 / fs)
    want = [2 * math.pi * f * math.cos(2 * math.pi * f * ti) for ti in t]
    err = max(abs(dx[i] - want[i]) for i in range(2, len(t) - 2))
    # 中心差分截断误差上界 = |x'''| dt^2/6 = (2*pi*f)^3 dt^2/6
    bound = (2 * math.pi * f) ** 3 * (1.0 / fs) ** 2 / 6.0
    assert err < bound * 1.05


def test_metrics_helpers():
    x = [1.0, 2.0, 3.0, 4.0]
    assert filters.mean(x) == 2.5
    assert filters.pstdev(x) == pytest.approx(math.sqrt(5.0 / 3.0))
    assert filters.quantile([0.0, 10.0], 0.5) == 5.0
    assert filters.pearson(x, [2.0, 4.0, 6.0, 8.0]) == pytest.approx(1.0)
    a, b, r = filters.linear_fit(x, [3.0, 5.0, 7.0, 9.0])
    assert (a, b) == pytest.approx((1.0, 2.0))
    assert r == pytest.approx(1.0)
    assert filters.r2_score(x, x) == 1.0


# ---------------------------------------------------------------------------
# Session I/O
# ---------------------------------------------------------------------------

def _jittered_session(n_samples=300, n_joints=7):
    out = []
    t = 0.0
    for k in range(n_samples):
        t += 0.01 if k % 3 else 0.013
        out.append(Sample(
            t=t, seq=k, mode=3, mode_name="POS", flags=0x200, joint_fault=0,
            q=[0.1 * math.sin(t) + 0.01 * j for j in range(n_joints)],
            dq=[0.1 * math.cos(t) for _ in range(n_joints)],
            tau=[0.2 * j for j in range(n_joints)],
            t_mos=[30.0 + j for j in range(n_joints)],
            t_coil=[31.0 + j for j in range(n_joints)],
            err=[0] * n_joints))
    return Session(out, {"fs": 100.0, "scenario": "unit"})


def test_session_resample_is_uniform(tmp_path):
    sess = _jittered_session()
    r = sess.resample(50.0)
    dts = [b.t - a.t for a, b in zip(r.samples, r.samples[1:])]
    assert max(abs(d - 0.02) for d in dts) < 1e-9
    assert r.n == 7
    paths = r.dump(str(tmp_path))
    assert os.path.exists(paths["status_csv"])
    back = Session.load(str(tmp_path))
    assert len(back.samples) == len(r.samples)
    assert back.samples[10].q[3] == pytest.approx(r.samples[10].q[3], abs=1e-9)


# ---------------------------------------------------------------------------
# Scenario safety
# ---------------------------------------------------------------------------

def test_scenario_limit_margin_is_enforced():
    limits = [(-3.0, 3.0)] * 7
    with pytest.raises(ValueError, match="余量"):
        validate_scenario(ScenarioSpec(kind="hold", q_center=[2.95] + [0.0] * 6,
                                       duration=1.0), limits)
    with pytest.raises(ValueError, match="amp"):
        validate_scenario(ScenarioSpec(kind="sine", q_center=[0.0] * 7,
                                       amp=0.0, freq=0.2, duration=1.0), limits)


def test_scenario_rejects_non_finite_times():
    limits = [(-3.0, 3.0)] * 7
    with pytest.raises(ValueError, match="有限"):
        validate_scenario(ScenarioSpec(kind="hold", q_center=[0.0] * 7,
                                       duration=float("nan")), limits)
    with pytest.raises(ValueError, match="有限"):
        validate_scenario(ScenarioSpec(kind="sine", q_center=[0.0] * 7,
                                       amp=0.1, freq=0.2,
                                       duration=float("inf")), limits)
    with pytest.raises(ValueError, match="有限"):
        validate_scenario(ScenarioSpec(kind="poses", q_center=[0.0] * 7,
                                       poses=[[0.0] * 7],
                                       dwell=float("nan")), limits)
    with pytest.raises(ValueError, match="有限"):
        validate_scenario(ScenarioSpec(kind="poses", q_center=[0.0] * 7,
                                       poses=[[0.0] * 7],
                                       dwell=1.0, pre_roll=float("inf")), limits)


def test_run_scenario_refuses_without_go():
    """go=False 必须无条件拒绝, 不能靠"臂未使能"兜底。"""
    spec = ScenarioSpec(kind="hold", q_center=[0.0] * 7, duration=1.0)
    with pytest.raises(RuntimeError, match="go=True"):
        run_scenario(object(), spec, "never-used", fs=100.0, go=False)


# ---------------------------------------------------------------------------
# Synthetic end-to-end: generator model == analyzer model
# ---------------------------------------------------------------------------

def _synthetic_session(fs=100.0, duration=2.0):
    snap = factory_snapshot()
    snap.firmware = "synthetic-test"
    samples = []
    amp, f = 0.15, 0.3
    for k in range(int(duration * fs) + 1):
        t = k / fs
        q, dq, ddq = [], [], []
        for j in range(7):
            ph = 2 * math.pi * f * t + 0.3 * j
            q.append(amp * math.sin(ph))
            dq.append(amp * 2 * math.pi * f * math.cos(ph))
            ddq.append(-amp * (2 * math.pi * f) ** 2 * math.sin(ph))
        d = rnea.theory_torques(snap, q, dq, ddq)
        samples.append(Sample(
            t=t, seq=k, mode=3, mode_name="POS", flags=0x200, joint_fault=0,
            q=q, dq=dq, tau=list(d["total"]),
            t_mos=[30.0] * 7, t_coil=[31.0] * 7, err=[0] * 7))
    return snap, Session(samples, {"fs": fs, "scenario": "sine"})


def test_analysis_and_outputs_offline(tmp_path):
    snap, sess = _synthetic_session()
    result = analyze_session(sess, snap, AnalyzeOptions())
    agg = result.summary["metrics"]["raw"]["aggregate"]
    assert agg["n_joints"] == 7
    assert agg["mean_rel_rmse"] < 1e-3      # 同一模型回放, 只应有数值噪声
    assert result.summary["diagnosis"][0]["source"] == "模型与实测高度一致"

    d = tmp_path / "session"
    paths = sess.dump(str(d))
    (d / "model.json").write_text(json.dumps(snap.to_dict()), encoding="utf-8")
    assert os.path.exists(paths["meta_json"])

    figs = generate_all(result, str(d / "figures"))
    assert len(figs) >= 20
    for p in figs.values():
        ET.parse(p)                          # 每个 SVG 都必须是合法 XML

    # x 轴必须覆盖整个绘图区 (曾因 xhi 取了首点的 y 值, 曲线被压到左边缘)
    svg = open(figs["j1_tau"], encoding="utf-8").read()
    xs = [float(m.group(1)) for m in re.finditer(r"[ML](-?[\d.]+),(-?[\d.]+)", svg)]
    assert min(xs) >= 77.5 and max(xs) <= 936.5
    assert max(xs) - min(xs) > 800

    report = write_report(result, figs, str(d), command="pytest")
    text = open(report, encoding="utf-8").read()
    assert "电流环力矩估计精度评估报告" in text
    assert "逐轴指标" in text
    assert "figures/joint1_tau.svg" in text


def test_analyze_validates_cutoffs_and_trim():
    snap, sess = _synthetic_session()
    with pytest.raises(ValueError, match="cutoff_ddq"):
        analyze_session(sess, snap, AnalyzeOptions(cutoff_ddq=-1.0))
    with pytest.raises(ValueError, match="cutoff_dq"):
        analyze_session(sess, snap, AnalyzeOptions(cutoff_dq=float("nan")))
    with pytest.raises(ValueError, match="trim_s"):
        analyze_session(sess, snap, AnalyzeOptions(trim_s=float("nan")))
    # 0 表示不滤波, 不能因边沿裁剪的 2/fc 除零而崩。
    result = analyze_session(sess, snap, AnalyzeOptions(cutoff_ddq=0.0))
    assert result.n_samples > 0


class _FakeResult:
    def __init__(self, scenario):
        self.summary = {
            "scenario": scenario,
            "n_samples": 10,
            "metrics": {"raw": {"aggregate": {
                "mean_rmse": 0.1, "max_rmse": 0.2,
                "worst_joint": "J1", "mean_abs_bias": 0.05}}},
        }


def test_overview_links_point_at_actual_subdirs(tmp_path):
    entries = [(_FakeResult("hold"), "01_hold"),
               (_FakeResult("sine"), "02_sine")]
    path = _write_overview(str(tmp_path), entries)
    text = open(path, encoding="utf-8").read()
    assert "(01_hold/report.md)" in text
    assert "(02_sine/report.md)" in text
    path = _write_overview(str(tmp_path), entries, no_report=True)
    assert "report.md" not in open(path, encoding="utf-8").read()


class _FakeArm:
    n = 7
    firmware = "fake"

    def get_state(self):
        return SimpleNamespace(value=SimpleNamespace(
            faulted=False, q=[0.0] * 7, dq=[0.0] * 7, tau=[0.0] * 7,
            mode_name="POS", enabled=True, flags=0))

    def close(self):
        pass


def test_collect_writes_one_directory_per_scenario(tmp_path, monkeypatch):
    """默认 all 场景不能互相覆盖 —— 每个场景一个子目录。"""
    recorded = []

    def fake_run(arm, spec, outdir, **kwargs):
        recorded.append(outdir)
        os.makedirs(outdir, exist_ok=True)

    monkeypatch.setattr(cli, "_connect", lambda args: _FakeArm())
    monkeypatch.setattr(cli, "read_joint_limits", lambda arm: [(-3.0, 3.0)] * 7)
    monkeypatch.setattr(cli, "fetch_from_arm", lambda arm: None)
    monkeypatch.setattr(cli, "run_scenario", fake_run)
    monkeypatch.setattr(cli, "_analyze_one",
                        lambda d, out, opts, cmd, no_plot, no_report:
                        _FakeResult(os.path.basename(d)))

    rc = cli.main(["collect", "--go", "--outdir", str(tmp_path)])
    assert rc == 0
    assert [os.path.basename(p) for p in recorded] == \
        ["01_hold", "02_sine", "03_poses"]
    assert all(os.path.isdir(p) for p in recorded)
    overview = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "(01_hold/report.md)" in overview
    assert "(03_poses/report.md)" in overview


def test_report_flags_friction_model_and_read_warnings(tmp_path):
    snap, sess = _synthetic_session()
    result = analyze_session(sess, snap, AnalyzeOptions())
    result.summary["model"]["friction_model"] = 2
    result.summary["model"]["read_warnings"] = ["get_body 失败"]
    path = write_report(result, {}, str(tmp_path), command="pytest")
    text = open(path, encoding="utf-8").read()
    assert "摩擦项未纳入前向计算" in text
    assert "模型读回警告: get_body 失败" in text
