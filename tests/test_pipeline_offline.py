"""离线端到端 —— 用 core 的桩传输把整条管线跑一遍。

⚠ 桩**不建模运动** ⇒ 运动类指标的读数是退化的。本测试的要点正是：
**退化读数绝不能被当成好成绩**（"什么都没测却得 78 分"是最危险的假绿）。
"""

import json
import os

import pytest

from litearm_benchmark import run
from litearm_benchmark.metrics.registry import ALL_METRICS
from litearm_benchmark.replay import compare, rescore


def test_offline_quick_tier_produces_no_total(tmp_path):
    out = str(tmp_path / "run")
    res = run("quick", dry_run=True, outdir=out, interactive=False)

    # ① 仿真绝不出总分
    assert res.card.simulated is True
    assert res.card.total is None, "仿真竟然出了总分"
    assert res.card.grade == "—"

    # ② **负控**：`total is None` 也可能是因为分析静默地什么都没算出来。
    #    所以必须同时证明"原始值其实算出来了不少" —— 否则这条判据是空转的。
    n_vals = len(res.raw.values)
    assert n_vals >= len(ALL_METRICS) - 2, \
        f"只算出 {n_vals} 个原始值（共 {len(ALL_METRICS)} 项）⇒ 管线可能是坏的，而不是被闸门挡住"
    assert res.raw.values.get("static.err_max_mrad") is not None, "连静差都没算出来"

    # ③ 三层落盘齐全
    for f in ("trace.npz", "raw.json", "metrics.csv", "REPORT.md"):
        assert os.path.exists(os.path.join(out, f)), f"缺 {f}"
    md = open(os.path.join(out, "REPORT.md"), encoding="utf-8").read()
    assert "不出总分" in md, "报告里没有仿真横幅"


def test_run_records_identity_and_excitation(tmp_path):
    out = str(tmp_path / "run")
    res = run("quick", dry_run=True, outdir=out, interactive=False)
    ident = json.load(open(os.path.join(out, "raw.json"), encoding="utf-8"))["identity"]
    # 机器身份是分数可比的前提 —— 缺了它分数不可解释
    for k in ("firmware", "n_joints", "ff_mask", "joint_params"):
        assert k in ident, f"身份里缺 {k}"
    assert "_excitation" in res.raw.evidence


def test_rescore_is_consistent_with_the_live_run(tmp_path):
    """重放必须与当时跑出来的结果一致 —— 否则"改权重不动硬件"这条承诺是假的。"""
    out = str(tmp_path / "run")
    live = run("quick", dry_run=True, outdir=out, interactive=False)
    again = rescore(out, write_report=False)
    assert again.simulated == live.card.simulated
    assert again.total == live.card.total
    for m in ALL_METRICS:
        a, b = live.card.metric(m.key), again.metric(m.key)
        assert a is not None and b is not None, f"{m.key} 重放后丢了"
        assert a.score == b.score, f"{m.key} 重放后分数变了：{a.score} → {b.score}"


def test_compare_runs_and_reports_data_gaps(tmp_path):
    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    run("quick", dry_run=True, outdir=a, interactive=False)
    run("quick", dry_run=True, outdir=b, interactive=False)
    md = compare(a, b)
    assert "总分" in md and "逐指标" in md
    # 仿真两次都没有总分 ⇒ 对比表里两侧都应是 "—"
    assert "| A | — |" in md and "| B | — |" in md


def test_manual_tier_refuses_to_run_non_interactively():
    """含人工动作的档位在非交互模式下必须**拒绝**，而不是静默跳过那些动作。"""
    with pytest.raises(RuntimeError, match="人工"):
        run("manual_load", dry_run=True, interactive=False, outdir=None)
