"""litearm-benchmark —— LiteArm 机械臂的**综合性能基准**。

它回答一个问题：**这台臂现在整体有多好？** —— 像 GPU 跑分那样，把位置跟踪精度、
速度环跟踪、力矩、抖动、笛卡尔、同步、热、耐久、安全、固件计算性能、链路质量
这些维度各自打分，再加权出一个 0~100 的总分与等级。

定位与边界
----------

* **建立在 `litearm-core` 之上**：只走 core 的公开面（CDC 直连固件），不碰 CAN、
  不依赖 `pylitearm`、不带自己的运动学模型。取不到的量（母线电流、任意构型 FK、
  在途笛卡尔进度…）在 :mod:`.instruments` 里**显式登记**，相应指标标「未实测」。
* **不产生假分**：测不出来就是「未实测」并**不参与**总分聚合；判据自证不达标就
  **整组剔除**。这两条是本工程与"随便量量"的分界线。
* **口径全部可追溯**：每一个阈值都在 :mod:`.metrics.registry` 里带 `provenance`，
  说明它是实测来的、物理推出的、还是占位。

典型用法::

    from litearm_benchmark import run, rescore, compare

    res = run("quick", outdir="docs/runA")      # ~5 分钟，会动真机
    print(res.summary())                        # "87.3/100 等级 A ..."

    rescore("docs/runA")                        # 改权重/阈值后重算，不动硬件
    print(compare("docs/runA", "docs/runB"))    # 逐指标对比

⚠ **会动真机**。CLI 上必须显式 `--go`；库用法请确保臂在安全台架包线内。
"""

#: ⚠ 必须在任何子模块 import **之前**定义：`report` / `session` / `cli` 都会
#: `from . import __version__`，放后面就是循环 import（cyc 已踩）。
__version__ = "0.1.0"

from litearm_benchmark import instruments
from litearm_benchmark.metrics.registry import (
    ALL_METRICS, DIMENSION_ORDER, DIMENSION_WEIGHTS, Direction, MetricSpec,
)
from litearm_benchmark.capture import DualCapture, Trace, snapshot_identity
from litearm_benchmark.align import AlignResult, align
from litearm_benchmark.metrics.analysis import RawResult, Segment, analyze
from litearm_benchmark.score import ScoreCard, score
from litearm_benchmark.replay import compare, load_raw, rescore
from litearm_benchmark.report import render_markdown, save_run, write_csv
from litearm_benchmark.safety import PreflightReport, SafetyAbort, park, preflight
from litearm_benchmark.session import BenchSession, RunResult, run
from litearm_benchmark.suites import tiers

__all__ = [
    # 顶层编排
    "run", "BenchSession", "RunResult",
    # 会话与安全
    "preflight", "park", "PreflightReport", "SafetyAbort",
    # 采集与分析
    "DualCapture", "Trace", "snapshot_identity", "Segment", "RawResult", "analyze",
    "align", "AlignResult",
    # 打分与报告
    "score", "ScoreCard", "render_markdown", "save_run", "write_csv",
    "rescore", "compare", "load_raw",
    # 目录与口径
    "instruments", "tiers",
    "ALL_METRICS", "DIMENSION_ORDER", "DIMENSION_WEIGHTS", "MetricSpec", "Direction",
    "__version__",
]
