#!/usr/bin/env python3
"""从指标目录生成 `docs/METRICS.md`。

**不要手改 `docs/METRICS.md`** —— 它是注册表的投影，改了会在下次生成时丢掉。
改指标请改 `src/litearm_benchmark/metrics/registry.py`，然后重跑本脚本。

    python3 scripts/gen_metrics_doc.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from litearm_benchmark import instruments as I                      # noqa: E402
from litearm_benchmark.metrics.registry import (                    # noqa: E402
    ALL_METRICS, DIMENSION_ORDER, DIMENSION_WEIGHTS, Direction,
)

HEAD = """# 指标字典

> ⚠ **本文件由 `scripts/gen_metrics_doc.py` 从指标注册表生成，不要手改。**
> 改指标请改 `src/litearm_benchmark/metrics/registry.py` 后重跑该脚本。

每个指标的阈值曲线都带 **出处**：是实测锚定的、物理推出的，还是**占位**（待标定）。
把占位当实测用，是这套体系里最容易犯、也最难发现的错。

"""


def main() -> int:
    L = [HEAD]
    L.append(f"共 **{len(ALL_METRICS)}** 项指标，分 **{len(DIMENSION_ORDER)}** 个维度。\n")
    L.append("| 维度 | 权重 | 指标数 |\n|---|---:|---:|")
    for d in DIMENSION_ORDER:
        n = sum(1 for m in ALL_METRICS if m.dimension == d)
        if n:
            L.append(f"| {d} | {DIMENSION_WEIGHTS.get(d, 1.0):.1f} | {n} |")
    L.append("")

    for d in DIMENSION_ORDER:
        ms = [m for m in ALL_METRICS if m.dimension == d]
        if not ms:
            continue
        L.append(f"## {d}\n")
        for m in ms:
            arrow = "越小越好" if m.direction is Direction.LOWER_BETTER else (
                "越大越好" if m.direction is Direction.HIGHER_BETTER else "落在带内最好")
            L.append(f"### `{m.key}`\n")
            L.append(f"- 单位：`{m.unit}`　方向：**{arrow}**　维度内权重：{m.weight:g}")
            L.append(f"- 分数曲线：`{m.curve_text()}`")
            if m.requires:
                L.append(f"- 需要：{', '.join('`' + r + '`' for r in m.requires)}")
            L.append(f"- **出处**：{m.provenance}")
            if m.note:
                L.append(f"- 说明：{m.note}")
            L.append("")

    L.append("## 仪表边界\n")
    L.append(I.describe())
    L.append("")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "METRICS.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print(f"已生成 {os.path.normpath(out)}（{len(ALL_METRICS)} 项指标）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
