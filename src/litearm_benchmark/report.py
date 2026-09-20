"""报告与落盘 —— 成绩单（Markdown）+ 明细（CSV）+ 原始数据（npz/json）。

落盘格式刻意分成三层，因为它们**寿命不同**：

* ``trace.npz`` —— 原始双流。**最贵**（要动硬件），必须一次存全，之后永不再采。
* ``raw.json``   —— 算出来的原始值 + 每条的证据与说明。改**权重/阈值**时只要它。
* ``REPORT.md``  —— 人看的。每次 rescore 都重生成，**不手改**。

⚠ 报告里每个数都必须能回答"哪来的"：所以明细表带 `证据` 列，仪表边界单独一节，
并且 **「未实测」与「不可信」是两个不同的标记**（前者没数据，后者有数据但判据不可信）。
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

from . import __version__
from .metrics.analysis import RawResult
from .score import ScoreCard


# --------------------------------------------------------------------------- #
def render_markdown(card: ScoreCard, raw: RawResult, *,
                    title: str = "LiteArm 基准测试报告",
                    tier: str = "", when: str = "") -> str:
    L = [f"# {title}", ""]
    meta = [f"版本 `{__version__}`"]
    if tier:
        meta.append(f"档位 **{tier}**")
    if when:
        meta.append(when)
    L.append("　".join(meta))
    L.append("")

    if card.simulated:
        L += ["> ⛔⛔ **本次是 dry-run / 仿真数据，不出总分。** 桩传输不建模运动，"
              "运动类指标的退化读数（抖动 0、τ 恒 0）会**伪装成满分**。\n"
              "> 要分数请接真机重跑。", ""]
    ident = card.identity or {}
    L += ["## 机器身份", "", "| 项 | 值 |", "|---|---|"]
    for k in ("firmware", "n_joints", "ff_mask", "hold_kp_gain"):
        if k in ident:
            v = ident[k]
            L.append(f"| `{k}` | {hex(v) if k == 'ff_mask' and isinstance(v, int) else v} |")
    if ident.get("joint_params"):
        L.append("| 逐轴 kp/kd/tau_max | "
                 + ", ".join(f"J{i+1}:{p['kp']:.0f}/{p['kd']:.0f}/{p['tau_max']:.0f}"
                             for i, p in enumerate(ident["joint_params"])) + " |")
    L.append("")

    t = card.total
    L += ["## 总分", ""]
    if t is None:
        L += ["> ⛔ **本次没有任何可计分的维度** —— 不产生总分。", ""]
    else:
        L += [f"### **{t:.1f} / 100　等级 **{card.grade}**", "",
              f"（由 {len(card.scored_dimensions)} 个已测维度加权得到；"
              f"未测维度见下节，**不参与**且不从 0 分起算）", ""]

    L += ["## 维度得分", "", "| 维度 | 权重 | 得分 | 已计分指标 |", "|---|---:|---:|---|"]
    for d in card.dimensions:
        s = "—" if d.score is None else f"{d.score:.1f}"
        L.append(f"| {d.name} | {d.weight:.1f} | {s} | {d.measured_ratio} |")
    L.append("")
    if card.missing_dimensions:
        L += ["**未参与总分的维度**：", ""]
        for name in card.missing_dimensions:
            L.append(f"- {name}：{card.notes.get('dim:' + name, '全部未实测')}")
        L.append("")

    L += ["## 指标明细", ""]
    for d in card.dimensions:
        L.append(f"### {d.name}　（{('%.1f' % d.score) if d.score is not None else '—'}）")
        L.append("")
        L += ["| 指标 | 原始值 | 单位 | 分 | 阈值曲线 | 证据 / 说明 |",
              "|---|---:|---|---:|---|---|"]
        for m in d.metrics:
            sp = m.spec
            if not m.trusted:
                vstr, sstr = (f"{m.raw:g}" if m.raw is not None else "—"), "剔除"
                note = f"⛔ {m.untrusted_why}"
            elif m.score is None:
                vstr, sstr = "—", "未实测"
                note = m.note or "本档未采到该指标所需的数据"
            else:
                vstr, sstr = f"{m.raw:g}", f"{m.score:.0f}"
                note = (m.evidence or "") + (f"　⚠ {m.note}" if m.note else "")
            L.append(f"| `{sp.key}` | {vstr} | {sp.unit} | {sstr} | "
                     f"{sp.curve_text()} | {note} |")
        L.append("")

    # 未实测 / 剔除 汇总（单列，避免被埋在明细里）
    # ⚠ 排除已被判"不可信"的：它们会同时出现在下面两节里，读者会以为是两回事
    missing = [m for d in card.dimensions for m in d.metrics
               if m.score is None and m.trusted]
    untrusted = [m for d in card.dimensions for m in d.metrics if not m.trusted]
    if untrusted:
        L += ["## ⛔ 被判为「不可信」而剔除的指标", ""]
        for m in untrusted:
            L.append(f"- `{m.spec.key}`（原始值 {m.raw:g}）：{m.untrusted_why}" if m.raw is not None
                     else f"- `{m.spec.key}`：{m.untrusted_why}")
        L.append("")
    if missing:
        L += ["## 未实测指标", "", f"共 {len(missing)} 项（**不是 0 分**，是没测）：", ""]
        for m in missing:
            L.append(f"- `{m.spec.key}`：{m.note or '本档未采到所需数据'}")
        L.append("")

    from . import instruments as I
    L += ["## 仪表边界（所有阈值的口径依据）", "", I.describe(), ""]
    ph = [k for k in missing if False]
    from .metrics.registry import unresolved_placeholders
    placeholders = unresolved_placeholders()
    if placeholders:
        L += ["## ⚠ 仍靠「占位」阈值的指标（待真机标定）", "",
              "这些指标的锚点**尚未实测标定**，分数只有相对意义：", ""]
        for k in placeholders:
            L.append(f"- `{k}`")
        L.append("")

    L += ["---", "",
          f"生成自 `litearm-benchmark {__version__}`。本文件可随时由 `raw.json` 重算，"
          "**不要手改**。", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def write_csv(card: ScoreCard, path: str) -> None:
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dimension", "dim_weight", "dim_score", "metric", "raw", "unit",
                    "score", "trusted", "requires", "provenance", "evidence"])
        for d in card.dimensions:
            for m in d.metrics:
                w.writerow([d.name, d.weight, "" if d.score is None else f"{d.score:.2f}",
                            m.spec.key, "" if m.raw is None else f"{m.raw:g}", m.spec.unit,
                            "" if m.score is None else f"{m.score:.1f}",
                            int(m.trusted), "|".join(m.spec.requires),
                            m.spec.provenance, m.evidence or m.note])


def save_run(outdir: str, *, trace, raw: RawResult, card: ScoreCard,
             extra: Optional[Dict[str, object]] = None) -> str:
    """落盘一次运行的三层数据。返回目录路径。"""
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(
        os.path.join(outdir, "trace.npz"),
        t_status=trace.t_status, q=trace.q, dq=trace.dq, tau=trace.tau,
        t_mos=trace.t_mos, t_coil=trace.t_coil, err=trace.err,
        flags=trace.flags, seq=trace.seq, mode=trace.mode,
        tcp_t=trace.tcp_t, tcp_p=trace.tcp_p,
        **{f"log{i}_tick": c.tick for i, c in enumerate(trace.chunks)},
        **{f"log{i}_qref": c.q_ref for i, c in enumerate(trace.chunks)},
        **{f"log{i}_tau": c.tau for i, c in enumerate(trace.chunks)},
    )
    with open(os.path.join(outdir, "raw.json"), "w", encoding="utf-8") as fh:
        json.dump({"version": __version__, "raw": raw.values,
                   "evidence": raw.evidence, "notes": raw.notes,
                   "identity": card.identity, "meta": trace.meta,
                   "marks": trace.marks, "extra": extra or {}},
                  fh, ensure_ascii=False, indent=2, default=str)
    open(os.path.join(outdir, "REPORT.md"), "w", encoding="utf-8").write(
        render_markdown(card, raw))
    write_csv(card, os.path.join(outdir, "metrics.csv"))
    return outdir
