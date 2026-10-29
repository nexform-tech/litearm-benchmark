"""离线重放 —— 拿已落盘的数据**重算分数**，不动硬件。

这是本工程里"最便宜的一件事、但价值最高"的部分：

* 改了**权重**或**阈值曲线** ⇒ 不必重跑 15 分钟的臂，`rescore` 一秒出结果；
* 换了**固件/参数** ⇒ 两次 run 的 `raw.json` 可以逐指标对比，看回归到底在哪一项上；
* 判据本身的**回归**：某个指标忽然全体变差，先怀疑指标实现而不是机器。

⚠ 重放只能重算**分数**，不能重算**原始值** —— 原始值依赖采集到的信号。
所以 `trace.npz` 必须一次存全（双流 + TCP + 日志窗口），后续任何分析都不该再碰硬件。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from .metrics.analysis import RawResult
from .score import ScoreCard, score


def load_raw(run_dir: str) -> Tuple[RawResult, Dict[str, object], Dict[str, object]]:
    """读回一次运行的 ``raw.json``。返回 ``(RawResult, identity, meta)``。"""
    path = os.path.join(run_dir, "raw.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 不存在 —— 该目录不是一次完整的 benchmark run")
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    raw = RawResult(values=dict(d.get("raw") or {}),
                    notes=dict(d.get("notes") or {}),
                    evidence=dict(d.get("evidence") or {}))
    return raw, dict(d.get("identity") or {}), dict(d.get("meta") or {})


def rescore(run_dir: str, *, write_report: bool = True) -> ScoreCard:
    """用**当前的** registry 重算这次运行的成绩单。"""
    raw, ident, meta = load_raw(run_dir)
    card = score(raw, identity=ident, simulated=bool(meta.get("simulated")))
    if write_report:
        from .report import render_markdown, write_csv
        open(os.path.join(run_dir, "REPORT.md"), "w", encoding="utf-8").write(
            render_markdown(card, raw))
        write_csv(card, os.path.join(run_dir, "metrics.csv"))
    return card


def compare(dir_a: str, dir_b: str, *, label_a: str = "A", label_b: str = "B",
            only_measured: bool = True) -> str:
    """两次运行的**逐指标**对比表（Markdown）。

    符号约定：`Δ` 一律是 ``B − A`` 的**原始值**差。因为各指标方向相反，
    这里**同时**给出两次的分数 —— 看分比看原始值差安全（本项目出过"越界峰写成负数当好消息"的事故）。
    """
    ra, ia, ma = load_raw(dir_a)
    rb, ib, mb = load_raw(dir_b)
    ca = score(ra, identity=ia, simulated=bool(ma.get("simulated")))
    cb = score(rb, identity=ib, simulated=bool(mb.get("simulated")))

    keys = sorted(set(ra.values) | set(rb.values))
    lines = [f"# 基准对比：{label_a} → {label_b}", "",
             f"- {label_a}: `{os.path.abspath(dir_a)}`",
             f"- {label_b}: `{os.path.abspath(dir_b)}`", ""]
    ta, tb = ca.total, cb.total
    fa = "—" if ta is None else f"{ta:.1f} ({ca.grade})"
    fb = "—" if tb is None else f"{tb:.1f} ({cb.grade})"
    lines += ["| | 总分 |", "|---|---:|", f"| {label_a} | {fa} |", f"| {label_b} | {fb} |", ""]
    if ta is not None and tb is not None:
        lines += [f"**总分变化 {tb - ta:+.1f}**", ""]

    lines += ["## 逐指标", "", "| 指标 | 单位 | " + label_a + " | " + label_b +
              " | Δ(原始) | 分(A→B) | 判读 |", "|---|---|---:|---:|---:|---|---|"]
    for k in keys:
        va, vb = ra.values.get(k), rb.values.get(k)
        ma, mb = ca.metric(k), cb.metric(k)
        if only_measured and va is None and vb is None:
            continue
        spec = (ma or mb).spec if (ma or mb) else None
        unit = spec.unit if spec else ""
        sa = "" if not ma or ma.score is None else f"{ma.score:.0f}"
        sb = "" if not mb or mb.score is None else f"{mb.score:.0f}"
        d = "" if (va is None or vb is None) else f"{vb - va:+g}"
        verdict = _verdict(va, vb, ma, mb)
        lines.append(f"| `{k}` | {unit} | {'—' if va is None else f'{va:g}'} | "
                     f"{'—' if vb is None else f'{vb:g}'} | {d} | {sa}→{sb} | {verdict} |")
    lines += ["", "> Δ 是原始值之差；判读按**分数**的方向给（各指标方向不同，别只看 Δ 的符号）。", ""]
    return "\n".join(lines)


def _verdict(va, vb, ma, mb) -> str:
    if va is None or vb is None:
        return "数据不全，不可比"
    if ma is None or mb is None or ma.score is None or mb.score is None:
        return "无分数"
    if not (ma.trusted and mb.trusted):
        return "⛔ 至少一侧不可信"
    d = mb.score - ma.score
    if abs(d) < 1.0:
        return "≈ 持平"
    return ("✅ 改善" if d > 0 else "❌ 变差") + f" {abs(d):.0f} 分"
