"""指标目录 —— **每个指标的分数曲线与阈值出处**。

本模块是「什么算好」的唯一声明处。分析模块（:mod:`.analysis` 与各 ``*_m.py``）
只负责**算出原始值**，从不算分；打分与阈值一律在这里。

⚠ 三条纪律（本项目血泪换来的）：

1. **每个锚点必须能回答"哪来的"**。`provenance` 里写清是实测（附量级与日期）、
   物理下界（附推导）、还是**占位**。占位不是耻辱，把占位当实测才是。
2. **方向要写对**。本项目历史上出过多次"把越界峰写成 −1.18 mrad 当成好成绩"
   这类符号事故 —— `Direction` 是强制项，不是注释。
3. **量程要可达**。锚点若全落在物理下界之下，指标**恒不可达**，比不设指标更糟。
   每个曲线的端点都对着 :mod:`..instruments` 里的地板数字核过。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple


class Direction(str, Enum):
    LOWER_BETTER = "lower"      # 越小越好
    HIGHER_BETTER = "higher"    # 越大越好
    BAND = "band"               # 落在 [lo, hi] 里最好（锚点仍需单调给出）


@dataclass(frozen=True)
class MetricSpec:
    """一个指标的元数据 + 分数曲线。"""

    key: str
    dimension: str
    unit: str
    direction: Direction
    anchors: Tuple[Tuple[float, float], ...]   # ((值, 分), ...)，**按值升序**
    provenance: str
    requires: Tuple[str, ...] = ()             # 需要什么才能测（见 REQUIRE_*）
    note: str = ""
    weight: float = 1.0                        # 维度内权重

    def score(self, value: Optional[float]) -> Optional[float]:
        """把原始值映射成 0~100。``None``/非有限 ⇒ ``None``（**不是 0** —— 未测 ≠ 得零分）。"""
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if v != v or v in (float("inf"), float("-inf")):
            return None
        pts = list(self.anchors)
        if self.direction is Direction.HIGHER_BETTER:
            pts = [(a, b) for a, b in pts]
        # 锚点按值升序；分数在该序下可升可降，线性插值即可（方向已体现在分数里）
        if v <= pts[0][0]:
            return float(pts[0][1])
        if v >= pts[-1][0]:
            return float(pts[-1][1])
        for (v0, s0), (v1, s1) in zip(pts, pts[1:]):
            if v0 <= v <= v1:
                if v1 == v0:
                    return float(max(s0, s1))
                t = (v - v0) / (v1 - v0)
                return float(s0 + t * (s1 - s0))
        return float(pts[-1][1])

    def curve_text(self) -> str:
        return "  ".join(f"{v:g}→{s:g}" for v, s in self.anchors)


# --------------------------------------------------------------------------- #
# `requires` 标签 —— 决定「没这门数据时标未实测而不是编分」
# --------------------------------------------------------------------------- #

# ⚠⚠ **这些标记目前只被 `report.py` 印进表里，不参与任何门控** —— 也就是说挂一个
#    `REQUIRE_MANUAL` **不会**让它在无人值守档里被跳过。别指望它有"拦"的作用。
#    （2026-09-21 复核：原本挂 `REQUIRE_MANUAL` 的三条**全都标错了** ——
#     `thermal.rise_per_min` 在无人值守的 `standard` 里正常出数、`endur.reconnects`
#     只是个计数器、`endur.static_drift_mrad` 现在每档都有 `drift_ref` 采数据。
#     三个错标已改正。）
REQUIRE_HW = "hardware"        # 必须真机
REQUIRE_LOG = "log300"         # 需要 300Hz 日志窗口
REQUIRE_CART = "cartesian"     # 需要笛卡尔能力
REQUIRE_MANUAL = "manual"      # 需要人工介入（装配重/复位）—— **仅标注**
REQUIRE_MODEL = "model"        # 需要动力学模型（G(q) 可算）


def _m(*a, **kw) -> MetricSpec:
    return MetricSpec(*a, **kw)


# ============================ A. 静态精度与重复性 ============================ #
STATIC = "static"
_DIM_STATIC = "静态精度与重复性"

M_STATIC_ERR_MAX = _m(
    "static.err_max_mrad", _DIM_STATIC, "mrad", Direction.LOWER_BETTER,
    ((0.0, 100), (2.0, 95), (5.0, 85), (15.0, 60), (30.0, 25), (60.0, 0)),
    provenance=("实测锚定：2026-09-20 持位交接缺陷期间 J2 恒偏 **−12.6~−14.8 mrad**（判为缺陷）；"
                "同一台臂修复后 **+0.4 mrad**。⇒ 15 mrad 判 60 分、30 mrad 判 25 分。"),
    note="逐轴 `|落点均值 − 指令目标|` 的最大值；**基准必须用 clamp 后的目标**（目标越软限时固件会把锚点钳回限内，拿未钳目标比会假红）。",
)

M_STATIC_REPEAT = _m(
    "static.repeat_ptp_mrad", _DIM_STATIC, "mrad", Direction.LOWER_BETTER,
    ((0.1, 100), (0.8, 90), (2.0, 75), (5.0, 50), (12.0, 20), (25.0, 0)),
    provenance=("实测锚定：2026-09-18/20 同一台臂上 N 轮落点极差 **0.76~3.1 mrad**；"
                "位置编码器 LSB 本身 0.096 mrad ⇒ 低于 0.1 不可信。"),
    note="同一目标重复 N 轮的极差（**相对第 1 次落点**）。⚠ 它单独**不能**说明准确度 —— "
         "每次都同样地偏在目标外时，它是 0。必须与 `static.err_max_mrad` 一起读。",
)

M_STATIC_FLOOR_RATIO = _m(
    "static.floor_ratio", _DIM_STATIC, "×", Direction.LOWER_BETTER,
    ((1.0, 100), (1.3, 90), (2.0, 75), (4.0, 45), (8.0, 15), (15.0, 0)),
    provenance=("**物理下界**：持位态内置前馈全停 ⇒ 稳态只由 `G(q_hold)` 与 `kp_hold·e` 平衡，"
                "库仑摩擦最多把臂拖开 `max(|fc0|,|fc1|)/kp_hold`（推导，`instruments.static_error_floor_rad`）。"
                "⇒ 该比值 **< 1 不可能**，锚点 1.0=100 分是对的。"),
    requires=(REQUIRE_MODEL,),
    note="尺寸无关的**归一化**静差：`静差 / (fc0 / kp_hold)`。换机器/换固件后仍可比。",
)

# ============================== B. 位置跟踪 ================================= #
TRACK = "tracking"
_DIM_TRACK = "位置跟踪"

M_TRACK_RMS = _m(
    "track.rms_mrad", _DIM_TRACK, "mrad", Direction.LOWER_BETTER,
    ((0.0, 100), (1.0, 95), (3.0, 80), (8.0, 55), (20.0, 25), (40.0, 0)),
    provenance=("实测锚定：修复后整条动作电池实测 rms **1.6~8 mrad**；修复前同一批（对齐口径正确时）"
                "在 10~40 mrad 量级。⚠ 历史 CSV 里 30~133 mrad 那组是**对齐失效**的产物，不可用作锚点。"),
    requires=(REQUIRE_HW, REQUIRE_LOG),
    note="**跨流对齐后**的残差 rms（300Hz 参考 vs 100Hz 实测）。**必须同时看 `track.align_ratio`** —— "
         "自证量不达标时这条数不可信。",
)

M_TRACK_PEAK = _m(
    "track.peak_mrad", _DIM_TRACK, "mrad", Direction.LOWER_BETTER,
    ((0.0, 100), (5.0, 92), (15.0, 72), (40.0, 45), (90.0, 15), (180.0, 0)),
    provenance="同 `track.rms_mrad`（实测同批）。取残差绝对值峰值。",
    requires=(REQUIRE_HW, REQUIRE_LOG),
)

M_TRACK_ALIGN_RATIO = _m(
    "track.align_ratio", _DIM_TRACK, "×", Direction.HIGHER_BETTER,
    ((1.0, 0), (2.0, 55), (3.0, 80), (5.0, 95), (8.0, 100)),
    provenance=("**自证量**：故意把时延再错开 80ms 的残差 / 最优时延下的残差。"
                "实测真结构 **3~6×**，离线随机负控 **1.00×** ⇒ 门槛取 2.0。"),
    requires=(REQUIRE_HW, REQUIRE_LOG),
    note="**这条不是性能指标，是「上面两条跟跟踪有关的数可不可信」的闸门。** "
         "低于 2× 时报告会把 `track.rms_mrad` 标成「不可信」而不是给低分。",
)

M_TRACK_LAG = _m(
    "track.lag_ms", _DIM_TRACK, "ms", Direction.LOWER_BETTER,
    ((0.0, 100), (8.0, 90), (20.0, 70), (45.0, 40), (90.0, 10), (150.0, 0)),
    provenance=("实测：修复后的动作电池里对齐 δ 落在 **−8 ~ −30 ms**（负=实测领先参考）。"
                "⚠ 它把**传输时延 + 伺服滞后 + 对齐估计误差**混在一起，不是纯伺服带宽。"
                "且**顶到搜索窗边缘时必须判为无效**（历史事故）。"),
    requires=(REQUIRE_HW, REQUIRE_LOG),
    note="越小越好但**别过读**：它是复合量。",
)

M_TRACK_FOLLOW_P95 = _m(
    "track.follow_p95_mrad", _DIM_TRACK, "mrad", Direction.LOWER_BETTER,
    ((0.0, 100), (5.0, 92), (15.0, 75), (40.0, 45), (100.0, 15), (250.0, 0)),
    provenance=("锚定到固件的**保护阈值**：跟随即误差判据在 `J1/J2=0.35 rad`、J3/J4=0.30、"
                "J5-J7=0.25（`safety_check.c`），且需**连续 5 周期**才锁存 ⇒ 250 mrad（0.25 rad）"
                "落到 0 分，0.35 rad 会直接触发保护。"),
    requires=(REQUIRE_HW,),
    note="运动段 `|q − q_ref|` 的 p95。⚠ 直接用 100Hz 两流相减**有对齐误差**，"
         "故这里用 p95 而不是 max，且只在 `track.align_ratio` 达标时才报。",
)

# ============================== C. 动态响应 ================================= #
DYNAMIC = "dynamic"
_DIM_DYN = "动态响应"

M_DYN_OVERSHOOT = _m(
    "dyn.overshoot_mrad", _DIM_DYN, "mrad", Direction.LOWER_BETTER,
    ((0.0, 100), (5.0, 92), (15.0, 72), (40.0, 40), (80.0, 10), (150.0, 0)),
    provenance=("实测锚定：动作电池的越界峰 —— 中低速档 **5~10 mrad**、高速档（0.15 rad @ speed 1.0）"
                "**31~42 mrad**。"),
    requires=(REQUIRE_HW,),
    note="落点越过目标的最大量（对**指令目标**取，不是对第一次落点）。",
)

M_DYN_REBOUND = _m(
    "dyn.rebound_mrad", _DIM_DYN, "mrad", Direction.LOWER_BETTER,
    ((0.0, 100), (5.0, 92), (15.0, 72), (40.0, 40), (90.0, 10), (180.0, 0)),
    provenance=("实测：回退幅度**恒 ≥ 0**（峰值后往回收的量）。中低速 0~7 mrad，"
                "高速档实测到 **62~68 mrad**。"),
    requires=(REQUIRE_HW,),
    note="越峰之后**回退**的幅度。⚠ 与「越峰」是两个量，历史上混过一次。",
)

M_DYN_SETTLE = _m(
    "dyn.settle_s", _DIM_DYN, "s", Direction.LOWER_BETTER,
    ((0.20, 100), (0.50, 88), (0.90, 72), (1.60, 45), (3.00, 15), (5.00, 0)),
    provenance=("实测：小位移档 0.5~1.0s、大行程多轴 2.2s。锚点按这批的分布取。"),
    requires=(REQUIRE_HW,),
    note="从发命令到进入并保持在到位带内的时间。",
)

M_DYN_RISE = _m(
    "dyn.rise_ms", _DIM_DYN, "ms", Direction.LOWER_BETTER,
    ((0.0, 100), (30.0, 92), (80.0, 75), (200.0, 45), (500.0, 15), (1000.0, 0)),
    provenance=("**占位** —— 锚点仍是量级估计。⚠ 两条容易踩的，都已写进实现："
                "（a）它随**指令速度**成比例变化 ⇒ **只能同档比**，故本指标只报"
                "**最快档**（旧实现跨档取一个中位数，量到的主要是「命令速度有多慢」）；"
                "（b）它同样随**行程距离**变化 ⇒ 同档内还要求行程相近。"
                "2026-09-21 真机 `full` 档实测各档：speed 0.2→1400 ms、0.3→680、0.4→800、"
                "0.5→1110（17 段，耐久循环）、0.6→940、0.8→439、**1.0→380 ms**。"
                "⚠ 注意它**不单调**（0.4 比 0.5 还快）—— 因为各档行程不同，"
                "不是一个单变量曲线。"
                "⛔⛔ **锚点的基础是错的，别把分数读成「响应慢」**：6 轮连续实测，"
                "同一档的上升时间极差只有 **0.26 ms（0.07%）** —— 说明它几乎完全由"
                "固件的 S 曲线参数 + 本次命令的速度/行程决定，与「臂好不好」关系不大。"
                "拿绝对 ms 做锚点，量的主要是「这次命令了多快多远」。"
                "**正确的锚定是相对指令剖面的比值**（实测 10→90% 时间 / `q_ref` 的 "
                "10→90% 时间：跟踪好 ⇒ ≈1，与速度、行程、机器都无关）—— "
                "本工程**尚未这么做**。"),
    requires=(REQUIRE_HW, REQUIRE_LOG),
)

# ============================ D. 平滑度与抖动 =============================== #
SMOOTH = "smoothness"
_DIM_SMOOTH = "平滑度与抖动"

M_SMOOTH_REST_SIGMA = _m(
    "smooth.rest_sigma_mrad", _DIM_SMOOTH, "mrad", Direction.LOWER_BETTER,
    ((0.05, 100), (0.20, 90), (0.50, 72), (1.20, 45), (2.50, 15), (5.00, 0)),
    provenance=("实测锚定：**对位置做直线拟合的残差 std** —— 根部 J1 ≈0.055 mrad（≈纯量化）、"
                "腕部 J6/J7 ≈0.13~0.17 mrad。⚠ 纯量化锯齿的理论 std = LSB/√12 = 0.028 mrad，"
                "实测腕部是它的 5~6 倍 ⇒ 真信号。"),
    requires=(REQUIRE_HW,),
    note="**静止**时的位置抖动。⚠ 必须用**直线拟合残差**，不能从单频分量反推"
         "（历史上这样低估过 3~4 倍）。",
)

M_SMOOTH_REST_RATIO = _m(
    "smooth.rest_vs_quant", _DIM_SMOOTH, "×", Direction.LOWER_BETTER,
    ((1.0, 100), (2.0, 88), (4.0, 70), (8.0, 40), (16.0, 10), (30.0, 0)),
    provenance=("**归一化**：抖动 std / (位置 LSB/√12)。纯量化锯齿 ⇒ 1.0。"
                "实测根部 ≈1（纯量化）、腕部 ≈5~6。"),
    requires=(REQUIRE_HW,),
    note="尺寸无关版；换编码器位数后仍可比。",
)

M_SMOOTH_VEL_RIPPLE = _m(
    "smooth.vel_ripple_pct", _DIM_SMOOTH, "%", Direction.LOWER_BETTER,
    ((0.0, 100), (1.0, 92), (4.0, 78), (10.0, 55), (20.0, 25), (40.0, 0)),
    provenance=("实测锚定：**巡航段**速度纹波 3%(J1/J2) ~ 13%(J6/J7) 的巡航值，且与运动相关、"
                "到位后归零。⚠ 判「纹波」前须扣掉上报 dq 的量化地板。"),
    requires=(REQUIRE_HW,),
    note="巡航段速度相对纹波 = std(v)/(|v|均值)。**必须取最长连续超阈段**切匀速段 —— "
         "用首末下标会被收尾那一拍参考尖峰把窗拖进减速段（历史上把纹波虚高过 12 倍）。",
)

M_SMOOTH_JERK_SPIKE = _m(
    "smooth.jerk_spike_ratio", _DIM_SMOOTH, "×", Direction.LOWER_BETTER,
    ((1.0, 100), (2.0, 90), (4.0, 72), (8.0, 45), (16.0, 15), (32.0, 0)),
    provenance=("**占位** —— 尚无多机型基线。定义：**巡航段之内** `max|Δv| / median|Δv|`"
                "（单流，不需要跨流对齐 ⇒ 无对齐伪影）。"
                "⚠ 2026-09-20 真机：1.0 档量到 **3.33**；而 0.5 档的巡航段内 "
                "`median|Δv|` **恰为 0**（参考在巡航段严格线性）⇒ **分母无定义**，"
                "那些段不计 —— 这是判据本身的**分辨力边界**，不是「很平滑」。"
                "⚠ 另：`|Δv|` 必须在**巡航段之内**取。旧实现取整个 chunk，"
                "而 chunk 里大部分是静止段（Δv≡0）⇒ 中位数被拉到 0 ⇒ **恒判未实测**。"),
    requires=(REQUIRE_HW,),
    note="「顿一下」的量化指纹：某一拍速度突跳。单流指标，最干净。"
         "⚠ 只反映**指令**剖面的平顺度（v 用 `q_ref` 差分 —— 日志的 `dq` 列在出厂 "
         "`ff_mask` 含 `FF_VELREF` 时被固件清零，恒 0 不可用）。",
)

M_SMOOTH_TAU_SIGMA = _m(
    "smooth.tau_sigma_nm", _DIM_SMOOTH, "N·m", Direction.LOWER_BETTER,
    ((0.0, 100), (0.02, 90), (0.06, 72), (0.15, 45), (0.40, 15), (1.00, 0)),
    provenance=("**归一化请用 `smooth.tau_vs_quant`**；本条给出绝对值，锚点按 J5-J7 的细 LSB "
                "(0.0048) 与 J1/J2 的粗 LSB (0.0586) 之间取。⚠ 判抖动前**必须先扣掉量化地板** "
                "`LSB/√12`，否则把台阶当抖动。"),
    requires=(REQUIRE_HW, REQUIRE_LOG),
    note="巡航段力矩 std，**已扣量化地板**（扣前会被 J1/J2 的粗 LSB 主导）。",
)

M_SMOOTH_TAU_RATIO = _m(
    "smooth.tau_vs_quant", _DIM_SMOOTH, "×", Direction.LOWER_BETTER,
    ((1.0, 100), (2.0, 88), (4.0, 68), (8.0, 38), (16.0, 10), (30.0, 0)),
    provenance="归一化版：力矩 std / (τ LSB/√12)。纯量化 ⇒ 1.0。",
    requires=(REQUIRE_HW, REQUIRE_LOG),
)

# ================================ E. 力矩 =================================== #
TORQUE = "torque"
_DIM_TORQUE = "力矩"

M_TAU_PEAK = _m(
    "torque.peak_ratio", _DIM_TORQUE, "×", Direction.LOWER_BETTER,
    ((0.05, 100), (0.20, 92), (0.40, 78), (0.65, 55), (0.85, 25), (1.00, 0)),
    provenance=("**物理锚定**：`tau_max` 是固件的硬钳位（`control_loop.c`），打到 1.0 就是饱和。"
                "实测动作电池的峰值占比在 0.05~0.25 之间，但**含过冲的档可到 0.65~0.75**。"),
    requires=(REQUIRE_HW,),
    note="`max|τ| / tau_max`（逐轴取最大）。",
)

M_TAU_SAT = _m(
    "torque.saturation_frac", _DIM_TORQUE, "×", Direction.LOWER_BETTER,
    ((0.0, 100), (0.005, 85), (0.02, 65), (0.08, 35), (0.20, 10), (0.40, 0)),
    provenance="**物理定义**：`|τ| ≥ 0.98·tau_max` 的采样占比。非零就意味着某段在饱和 —— 饱和会引入非线性，是控制品质下降的直接证据。",
    requires=(REQUIRE_HW,),
)

M_TAU_G_MATCH = _m(
    "torque.g_match_pct", _DIM_TORQUE, "%", Direction.LOWER_BETTER,
    ((0.0, 100), (4.0, 92), (8.0, 78), (15.0, 58), (26.0, 25), (50.0, 0)),
    provenance=("实测锚定：**静态**多姿态下 `τ_实测 vs G(q) 模型` 的偏差 —— 实测 **3.5% / 4.4% / "
                "6.3% / 10.9% / 25.2%**（均值≈10%、中位 6.3%）。锚点直接贴这批分布。"),
    requires=(REQUIRE_MODEL,),
    note="动力学模型的可信度。⚠ 必须在**稳态**读（装载/卸载的瞬态会把偏差读数高估一倍，历史上踩过）。",
)

# =============================== F. 笛卡尔 ================================== #
CART = "cartesian"
_DIM_CART = "笛卡尔"

M_CART_LINE = _m(
    "cart.line_dev_mm", _DIM_CART, "mm", Direction.LOWER_BETTER,
    ((0.3, 100), (1.0, 90), (2.0, 75), (4.0, 45), (8.0, 15), (15.0, 0)),
    provenance=("实测锚定：本项目实测直线径向偏差 **1.32~2.90 mm**，**且误差随 `|G(J2)|` 增大而增大**"
                "（负载相关）⇒ 锚点按同姿态同负载口径定。直线上**弦切走**的坏形状会造成 14.6 mm 量级。"),
    requires=(REQUIRE_HW, REQUIRE_CART),
    note="直线轨迹到理论直线的最大偏离。⚠ 比对精度**必须同姿态同负载**，否则数不可比。",
)

M_CART_ARC = _m(
    "cart.arc_dev_mm", _DIM_CART, "mm", Direction.LOWER_BETTER,
    ((0.3, 100), (1.2, 90), (2.5, 75), (5.0, 45), (9.0, 15), (16.0, 0)),
    provenance="同 `cart.line_dev_mm`（同一批实测的圆弧段）。",
    requires=(REQUIRE_HW, REQUIRE_CART),
)

M_CART_PLAN_US = _m(
    "cart.plan_us", _DIM_CART, "µs", Direction.LOWER_BETTER,
    ((200.0, 100), (1000.0, 90), (5000.0, 70), (20000.0, 40), (60000.0, 10), (120000.0, 0)),
    provenance=("实测：固件 `CartPlan.plan_us` 上报的规划耗时。整臂 7 轴的 IK+路径规划在 ms 量级。"
                "⚠ 锚点是**占位**量级估计，需按真机多路径实测收敛。"
                "⚠ 2026-09-20 真机（`standard` 档）：直线 50mm **4796 µs**、"
                "圆弧 r40 90° **5670 µs**，取最大值上报 —— **单点，未收敛**。"),
    requires=(REQUIRE_HW, REQUIRE_CART),
)

# ================================ G. 同步 =================================== #
SYNC = "sync"
_DIM_SYNC = "多轴同步"

M_SYNC_SPREAD = _m(
    "sync.stop_spread_ms", _DIM_SYNC, "ms", Direction.LOWER_BETTER,
    ((0.0, 100), (5.0, 92), (15.0, 78), (40.0, 55), (100.0, 20), (200.0, 0)),
    provenance=("⛔⛔ **本指标当前不给分（恒判未实测）—— 判据「构造即为真」，没有判别力。**"
                "`movej_sync` 由**一条标量 `s: 0→1` 同时驱动全轴**，core 的文档原文："
                "「保证是**参考级**同步（参考同拍到达）；**物理跟踪差异仍在**」"
                "⇒ 各轴**参考**速度必然同拍归零 ⇒ 量出来的极差**恒为 0**。"
                "2026-09-20 真机实测就是 **0.000 ms**，会恒判 100 分 —— 那是"
                "「验收只靠定义即真」，本项目明令禁止。"
                "⚠ 物理同步**测不了**：唯一含实测位置的是 100Hz 状态帧（10ms 分辨率），"
                "低于各轴停止分散的量级。⇒ 要恢复这一维，需要更快的实测流。"),
    requires=(REQUIRE_HW, REQUIRE_LOG),
    note="⛔ **恒未实测**（见 provenance）。保留它只为让报告里**看得见**「试过了、"
         "为什么给不出分」，而不是悄悄消失。当时的候选判据「全轴同乘一个速度系数」"
         "已被实测证伪，本判据（速度非零区间的**停止**分散）方向是对的，"
         "**但只能量到参考级**。",
)

# =============================== H. 热 ===================================== #
THERMAL = "thermal"
_DIM_THERMAL = "热"

M_THERMAL_MOS = _m(
    "thermal.t_mos_max", _DIM_THERMAL, "°C", Direction.LOWER_BETTER,
    ((30.0, 100), (45.0, 88), (60.0, 70), (75.0, 40), (90.0, 12), (110.0, 0)),
    provenance=("**器件锚定**：固件有温度预警通道（`temp_warning_cycles` 触发 `ARM_FLAG_TEMP_WARNING` "
                "并单轴锁存）⇒ 接近预警阈值就是坏。锚点按 MOS 常见降额点取。"),
    requires=(REQUIRE_HW,),
    note="全程 MOS 温度峰值。⚠ 温度只从 **100Hz 状态帧**取，日志里**没有**温度。",
)

M_THERMAL_RISE = _m(
    "thermal.rise_per_min", _DIM_THERMAL, "°C/min", Direction.LOWER_BETTER,
    ((0.0, 100), (0.5, 88), (1.5, 70), (3.0, 42), (6.0, 12), (12.0, 0)),
    provenance="**占位** —— 需要长时间（≥20min）运行才能拟出斜率；项目实测里长时间运行的热是平的。"
               "⚠ 2026-09-20 真机 `standard` 档量到 **−0.0 °C/min**（窗口太短，且热确实平）"
               "⇒ 这个数**说明不了稳定性**，要收敛必须上 `full` 档的 20min 热平衡窗口。",
    requires=(REQUIRE_HW,),
)

# ============================== I. 耐久 ==================================== #
ENDURANCE = "endurance"
_DIM_ENDUR = "耐久与稳定性"

M_ENDUR_SEQ = _m(
    "endur.seq_gap", _DIM_ENDUR, "帧", Direction.LOWER_BETTER,
    ((0.0, 100), (1.0, 80), (3.0, 55), (10.0, 25), (50.0, 5), (200.0, 0)),
    provenance=("**可算准**：状态帧 `seq` 是 16-bit 计数器（65536 帧 ≈ 11 min @100Hz），"
                "取 ``(seq₂−seq₁) mod 65536 − 1`` 即**两帧之间丢了几帧**。"
                "⚠⚠ **判据不许掺 `dt`**：固件的状态帧是**被动**的（主机不读它基本不发），"
                "所以 `dt × 100Hz` 量到的是**主机的读取节奏**而不是丢帧 —— 旧实现正是这么"
                "算的，真机实测把「一帧没丢」（`seq` 差 24383 / 收到 24384 帧）报成"
                "**37.46 帧、11 分**（假红）。"
                "⚠ 判据必须**同时**报出「有几对可比」—— 只有 0 对时不能判绿。"),
    requires=(REQUIRE_HW,),
    note="长时间运行中**固件丢帧**的量化证据。真机 10 次运行里 9 次为 **0**；"
         "唯一非 0 的一次是笛卡尔超时轮询那轮（91 处 `Δseq=2`）。",
)

M_ENDUR_RECONNECT = _m(
    "endur.reconnects", _DIM_ENDUR, "次", Direction.LOWER_BETTER,
    ((0.0, 100), (1.0, 40), (2.0, 15), (5.0, 0)),
    provenance="**期望恒 0**：一次长时间运行里不应出现任何意外重连。",
    requires=(REQUIRE_HW,),
)

M_ENDUR_DRIFT = _m(
    "endur.static_drift_mrad", _DIM_ENDUR, "mrad", Direction.LOWER_BETTER,
    ((0.0, 100), (1.0, 90), (3.0, 70), (8.0, 40), (20.0, 10), (40.0, 0)),
    provenance="**占位**：长时间运行前后**同一点位**的静差变化。它能把「热漂」「参数漂」「机械松动」三者混在一起 ⇒ 只作提示。"
               "⚠ 2026-09-20：`quick`/`standard` **跑不到它**（需要首尾同点位对照，只有 "
               "`full` 档的耐久循环才有）⇒ 至今**一次都没量到过**，锚点是纯占位。",
    requires=(REQUIRE_HW,),
)

# =============================== J. 安全 =================================== #
SAFETY = "safety"
_DIM_SAFETY = "安全与保护"

M_SAFE_WD = _m(
    "safe.watchdog_latency_s", _DIM_SAFETY, "s", Direction.LOWER_BETTER,
    ((0.10, 100), (0.20, 88), (0.35, 65), (0.60, 35), (1.00, 10), (2.00, 0)),
    provenance=("**设计锚定**：固件命令看门狗超时是 **100ms**（`mode==MOVE_J` 时每拍自动 kick），"
                "实测跳闸 **0.21~0.29 s**。⇒ 2.0s 判 0 分。"),
    requires=(REQUIRE_HW,),
    note="**停止发命令 → 观察到 fail-soft 生效**的时延。⚠ 判据要用 `flags & 0x2`（core 解成 "
         "`WD_TRIPPED`）——**按固件拼法搜 `WATCHDOG` 永远搜不到**（历史事故）。",
)

M_SAFE_MARGIN = _m(
    "safe.margin_rad", _DIM_SAFETY, "rad", Direction.HIGHER_BETTER,
    ((0.00, 0), (0.02, 25), (0.05, 55), (0.10, 75), (0.30, 92), (1.00, 100)),
    provenance=("**设计锚定**：固件的锁存线是 `软限 ± POS_LATCH_BAND`(0.05) "
                "(`safety_check.c:165`)，故 0 = 已在线**上**（下一拍就可能锁存）判 0 分；"
                "0.05 = 正好顶在软限（固件允许的指令边界，属正常工作位）判 55 分。"
                "⚠ **锚点于 2026-09-20 随口径修正重设**（原按「距软限」定，会把 J4 的"
                "正常零位判成 ~25 分）—— 新锚点**尚无真机分布，属待收敛**。"),
    requires=(REQUIRE_HW,),
    note="运行全程**最小**的距**锁存线**裕度（不是距软限 —— 见 "
         "`instruments.latch_margin_rad`）。⚠ 任何轴锁存 ⇒ 整臂静默降级且 `reset` 清不掉。",
)

M_SAFE_LATCH = _m(
    "safe.pos_violation_events", _DIM_SAFETY, "次", Direction.LOWER_BETTER,
    ((0.0, 100), (1.0, 40), (3.0, 12), (6.0, 0)),
    provenance="**期望恒 0**：位置越限锁存**不受 enabled 门控**、`reset` 也清不掉 ⇒ 出现即严重。",
    requires=(REQUIRE_HW,),
)

# =========================== K. 计算性能（固件） ============================ #
COMPUTE = "compute"
_DIM_COMPUTE = "固件计算性能"

M_COMPUTE_LOOP = _m(
    "compute.loop_max_kcycle", _DIM_COMPUTE, "kcycle", Direction.LOWER_BETTER,
    ((150.0, 100), (300.0, 92), (600.0, 75), (1000.0, 45), (1300.0, 15), (1600.0, 0)),
    provenance=("**硬件锚定**：控制拍周期 @300Hz = 480MHz/300 = **1600 kcycle**，"
                "峰值吃掉整个周期就必然丢拍 ⇒ 1600 判 0 分。"
                "⚠ **锚点于 2026-09-20 重设**：原锚点（20→100 … 240→0）是照 `LOOP1`"
                "——**禁中断单次**测的控制步计算量——定的，而本指标量的是"
                "**真实整拍峰值** `loop_k`，两者真机实测差 **10×**"
                "（`LOOP1` 20.06 vs `loop_k` 199 kcycle）。原锚点会把 199 判成约 5 分。"
                "新锚点在 1600 硬预算内取刻度，**中间各点尚无真机分布，属待收敛**。"),
    requires=(REQUIRE_HW,),
    note="固件 `loop_k`（`litearm.h:307`「历史最长单拍耗时」/1000），**不是** `LOOP1`。"
         "⚠ 它来自自检的第二帧（LINK 段）⇒ **只收到 1 帧时本指标判未实测**，不填 0。",
)

M_COMPUTE_OVR = _m(
    "compute.overruns", _DIM_COMPUTE, "拍", Direction.LOWER_BETTER,
    ((0.0, 100), (1.0, 60), (5.0, 25), (20.0, 5), (50.0, 0)),
    provenance="**期望恒 0**：控制拍超出预算的拍数。非 0 即控制周期被打破。",
    requires=(REQUIRE_HW,),
)

# =============================== L. 链路 =================================== #
LINK = "link"
_DIM_LINK = "通信链路"

M_LINK_STATUS_HZ = _m(
    "link.status_hz", _DIM_LINK, "Hz", Direction.HIGHER_BETTER,
    ((60.0, 0), (90.0, 50), (98.0, 85), (100.0, 92), (105.0, 100)),
    provenance=("**协议锚定**：被动状态帧固定 **100Hz** ⇒ 明显偏离 100 就是链路问题。"
                "实测 100.1Hz。高于 105 需要解释（重复计数？）。"),
    requires=(REQUIRE_HW,),
    note="实测状态帧到达率。⚠ 用 `Msg.hz`（core 的到达统计）或自己按时间戳算。",
)

M_LINK_CRC = _m(
    "link.crc_errors", _DIM_LINK, "次", Direction.LOWER_BETTER,
    ((0.0, 100), (1.0, 55), (5.0, 25), (20.0, 5), (100.0, 0)),
    provenance="**期望恒 0**：固件 LINK 计数的 CRC 错误数。",
    requires=(REQUIRE_HW,),
)

M_LINK_FOREIGN = _m(
    "link.foreign_frames", _DIM_LINK, "帧", Direction.LOWER_BETTER,
    ((0.0, 100), (1.0, 70), (10.0, 35), (100.0, 5), (1000.0, 0)),
    provenance=("实测事故：**广播抢帧**曾让读 RPC **21% 失败**（直连 0/200 vs 经 server 42/200）。"
                "⇒ 非 0 即说明有第二个会话在同一条链路上抢帧。"),
    requires=(REQUIRE_HW,),
)

# --------------------------------------------------------------------------- #
DIMENSION_ORDER = (
    _DIM_STATIC, _DIM_TRACK, _DIM_DYN, _DIM_SMOOTH, _DIM_TORQUE,
    _DIM_CART, _DIM_SYNC, _DIM_THERMAL, _DIM_ENDUR, _DIM_SAFETY,
    _DIM_COMPUTE, _DIM_LINK,
)

#: 维度权重（总分 = 加权平均，缺项维度会被**剔除并重新归一化**，报告里显式列出）。
DIMENSION_WEIGHTS: Dict[str, float] = {
    _DIM_STATIC: 1.6,     # 静态精度是「机械臂能不能干活」的地基
    _DIM_TRACK: 1.5,      # 跟踪
    _DIM_DYN: 1.0,        # 动态
    _DIM_SMOOTH: 1.2,     # 平滑 —— 直接影响手感与寿命
    _DIM_TORQUE: 1.0,
    _DIM_CART: 1.2,
    _DIM_SYNC: 0.8,
    _DIM_THERMAL: 0.8,
    _DIM_ENDUR: 1.0,
    _DIM_SAFETY: 1.5,     # 安全是底线
    _DIM_COMPUTE: 0.7,
    _DIM_LINK: 0.7,
}

ALL_METRICS: Tuple[MetricSpec, ...] = (
    M_STATIC_ERR_MAX, M_STATIC_REPEAT, M_STATIC_FLOOR_RATIO,
    M_TRACK_RMS, M_TRACK_PEAK, M_TRACK_ALIGN_RATIO, M_TRACK_LAG, M_TRACK_FOLLOW_P95,
    M_DYN_OVERSHOOT, M_DYN_REBOUND, M_DYN_SETTLE, M_DYN_RISE,
    M_SMOOTH_REST_SIGMA, M_SMOOTH_REST_RATIO, M_SMOOTH_VEL_RIPPLE,
    M_SMOOTH_JERK_SPIKE, M_SMOOTH_TAU_SIGMA, M_SMOOTH_TAU_RATIO,
    M_TAU_PEAK, M_TAU_SAT, M_TAU_G_MATCH,
    M_CART_LINE, M_CART_ARC, M_CART_PLAN_US,
    M_SYNC_SPREAD,
    M_THERMAL_MOS, M_THERMAL_RISE,
    M_ENDUR_SEQ, M_ENDUR_RECONNECT, M_ENDUR_DRIFT,
    M_SAFE_WD, M_SAFE_MARGIN, M_SAFE_LATCH,
    M_COMPUTE_LOOP, M_COMPUTE_OVR,
    M_LINK_STATUS_HZ, M_LINK_CRC, M_LINK_FOREIGN,
)

BY_KEY: Dict[str, MetricSpec] = {m.key: m for m in ALL_METRICS}


def unresolved_placeholders() -> list:
    """列出靠**占位**阈值的指标 —— 报告里要显眼，别忘了它们是待标定的。"""
    return [m.key for m in ALL_METRICS if "占位" in m.provenance]


def audit_curves() -> list:
    """自检：找出锚点不合规的指标。返回问题列表（空=全部合规）。

    抓三类错（都在本项目真实发生过）：
      * 锚点没按值升序 ⇒ 插值会算出非单调的分数；
      * 分数不在 [0,100]；
      * 单调性方向与 `direction` 矛盾（例如判 LOWER_BETTER 却给大值高分）。
    """
    bad = []
    for m in ALL_METRICS:
        vals = [v for v, _ in m.anchors]
        scs = [s for _, s in m.anchors]
        if any(b <= a for a, b in zip(vals, vals[1:])):
            bad.append(f"{m.key}: 锚点未按值严格升序 {vals}")
        if any(not (0 <= s <= 100) for s in scs):
            bad.append(f"{m.key}: 分数越界 {scs}")
        if m.direction is Direction.LOWER_BETTER and any(b > a for a, b in zip(scs, scs[1:])):
            bad.append(f"{m.key}: 声明 LOWER_BETTER 但分数随值上升 {scs}")
        if m.direction is Direction.HIGHER_BETTER and any(b < a for a, b in zip(scs, scs[1:])):
            bad.append(f"{m.key}: 声明 HIGHER_BETTER 但分数随值下降 {scs}")
    return bad
