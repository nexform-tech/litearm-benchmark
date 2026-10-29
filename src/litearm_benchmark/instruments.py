"""仪表边界 —— **本工程唯一的常数真相源**。

为什么单独一个模块：这些数字**每一个都能让指标算错一个量级**，而它们散落在
固件源码、记忆文件、历史实测报告里。把它们集中在这里、每条都带**出处与证据等级**，
是为了让「这个阈值/地板是哪来的」永远可回答。

⚠ 三条硬规矩（来自 2026-09 那一轮持位交接缺陷的复盘）：
  1. **指标模块不许硬编码量化/容量常数**，一律 `from .instruments import ...`。
  2. 任何「界」优先引**实测**；手推的界必须标 :attr:`Provenance.DERIVED` 并写明推导前提。
  3. 取不到的量要**显式登记**在 :data:`UNAVAILABLE`，而不是让指标静默返回 0。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class Provenance(str, Enum):
    """一个常数的证据等级 —— 报告里会逐条打印，别把 DERIVED 当成 MEASURED 用。"""

    MEASURED = "实测"        # 真机量出来的，附日期
    NOMINAL = "标称"         # 器件手册 / 编码器位数推出
    DERIVED = "推导"         # 由其它量算出，**写明前提**；前提变了它就失效
    PLACEHOLDER = "占位"     # 还没测，只是为了让管线跑起来 —— 报告里必须显眼


@dataclass(frozen=True)
class Constant:
    key: str
    value: object
    unit: str
    prov: Provenance
    note: str

    def __str__(self) -> str:  # pragma: no cover - 展示用
        return f"{self.key} = {self.value} {self.unit} [{self.prov.value}]"


# --------------------------------------------------------------------------- #
# 采样率与容量
# --------------------------------------------------------------------------- #

CTRL_HZ = Constant("CTRL_HZ", 300.0, "Hz", Provenance.MEASURED,
                   "固件控制拍。300Hz log 的拍率；`scurve.c` 与 `control_loop.c` 的 dt 由此定。")

STATUS_HZ = Constant("STATUS_HZ", 100.0, "Hz", Provenance.MEASURED,
                     "**被动状态帧固定 100Hz、不可配置**。实测 100.1Hz。"
                     "⇒ 实测 q/dq/tau/温度**只有 10ms 分辨率**，这是跟踪类指标的分辨率天花板。")

CTRL_DT = Constant("CTRL_DT", 1.0 / 300.0, "s", Provenance.DERIVED,
                   "= 1/CTRL_HZ。差分用 dt 必须用它，别写 0.0033。")

LOG_CAP_TICKS = Constant("LOG_CAP_TICKS", 2400, "拍", Provenance.MEASURED,
                         "固件 `LOG_MAX_SAMPLES`。n>2400 固件回 ERR{0x2D,0x02}。"
                         "⇒ **300Hz 日志一次最多 ~8.0s**，且记满自停 —— 不是连续流。")

LOG_CAP_S = Constant("LOG_CAP_S", 2400 / 300.0, "s", Provenance.DERIVED,
                     "= LOG_CAP_TICKS / CTRL_HZ ≈ 8.0s。长动作必须**分多次采集**。")


def log_sample_bytes(n_joints: int) -> int:
    """300Hz 日志单拍字节数 = `4 + 12N`（tick u32 + 每轴 q_ref/dq/tau 各 f32）。"""
    return 4 + 12 * n_joints


LOG_CHUNK_BYTES = Constant("LOG_CHUNK_BYTES", 60, "B", Provenance.MEASURED,
                           "`LogReader.iter_chunks` 每次往返的读块上限 ⇒ 满量程约 863 次往返。"
                           "⇒ 采集本身有可观的挂钟开销，做时长预算时要算进去。")


# --------------------------------------------------------------------------- #
# 各量的量化地板 —— **任何「抖动/纹波」判据必须先把地板扣掉**
# --------------------------------------------------------------------------- #

#: 300Hz 日志的 `tau` 与状态帧的 `tau` 都来自电机 12-bit 力矩反馈，LSB = 2·TMAX/4096。
#: ⚠ 这里的 TMAX 是**电机额定力矩**（DM 系列反馈量程），**不是**固件参数 `tau_max` ——
#: 两者的比值约 1.5，用错会把地板算小一半。
#: 出处：2026-09-18 真机实测（`stm32-log-trace-instrument-limits` 第 2 条）。
TAU_LSB_NM: Dict[int, Constant] = {
    1: Constant("TAU_LSB.J1", 0.0586, "N·m", Provenance.MEASURED, "DM6248P，2026-09-18"),
    2: Constant("TAU_LSB.J2", 0.0586, "N·m", Provenance.MEASURED, "DM6248P，2026-09-18"),
    3: Constant("TAU_LSB.J3", 0.0136, "N·m", Provenance.MEASURED, "DM4340，2026-09-18"),
    4: Constant("TAU_LSB.J4", 0.0136, "N·m", Provenance.MEASURED, "DM4340，2026-09-18"),
    5: Constant("TAU_LSB.J5", 0.0048, "N·m", Provenance.MEASURED, "DM4310，2026-09-18"),
    6: Constant("TAU_LSB.J6", 0.0048, "N·m", Provenance.MEASURED, "DM4310，2026-09-18"),
    7: Constant("TAU_LSB.J7", 0.0048, "N·m", Provenance.MEASURED, "DM4310，2026-09-18"),
}

#: 上报 `dq` 的量化：LSB = `limits[1] / 2048`，`limits` 是**逐台标定**的电机反馈量程，
#: **不在任何读回命令里**（0x24 只给 kp/kd/tau_max/q_min/q_max）⇒ 只能从固件
#: `defaults.c` 的 `.limits` 抄，并且**换电机/换批必须重抄**。
#: 出处：2026-09-18 真机实测（同记忆第 4 条）。
DQ_LSB_RAD_S: Dict[int, Constant] = {
    1: Constant("DQ_LSB.J1", 0.00977, "rad/s", Provenance.MEASURED, "limits[1]=20，2026-09-18"),
    2: Constant("DQ_LSB.J2", 0.00977, "rad/s", Provenance.MEASURED, "limits[1]=20"),
    3: Constant("DQ_LSB.J3", 0.00488, "rad/s", Provenance.MEASURED, "limits[1]=10"),
    4: Constant("DQ_LSB.J4", 0.00488, "rad/s", Provenance.MEASURED, "limits[1]=10"),
    5: Constant("DQ_LSB.J5", 0.00488, "rad/s", Provenance.MEASURED, "limits[1]=10"),
    6: Constant("DQ_LSB.J6", 0.01465, "rad/s", Provenance.MEASURED, "limits[1]=30（J6 个体）"),
    7: Constant("DQ_LSB.J7", 0.01465, "rad/s", Provenance.MEASURED, "limits[1]=30（J7 个体）"),
}

Q_LSB_RAD = Constant("Q_LSB_RAD", 2 * 3.141592653589793 / 65536.0, "rad", Provenance.NOMINAL,
                     "16-bit 编码器 ⇒ 9.59e-5 rad = 0.0959 mrad。"
                     "⚠ 上报 `dq` 还有**朝零截断**的系统偏差（幅值恒被低估 ≈0.5 LSB），"
                     "故「实测速度」应当**对位置差分**而不是用上报 dq。")

#: 上报 dq 的朝零截断偏差（幅值方向），慢速时相对误差可达 ~10%。
DQ_TRUNCATION_LSB = Constant("DQ_TRUNCATION_LSB", 0.5, "LSB", Provenance.MEASURED,
                             "2026-09-18 四段×七轴实测偏差/LSB 几乎全是 −0.45~−0.56。"
                             "⇒ 用上报 dq 算慢速跟踪会有系统性偏小。")

#: 力矩量化对「纹波 std」的贡献上限：均匀量化 std = LSB/√12。
def tau_quant_std_floor(nm_lsb: float) -> float:
    """力矩量化本身贡献的 std 地板 —— 判「力矩是否抖动」必须先减去它。"""
    return nm_lsb / 12 ** 0.5


# --------------------------------------------------------------------------- #
# 日志列语义（会决定「能不能用这一列」）
# --------------------------------------------------------------------------- #

LOG_COLUMNS = ("tick", "q_ref", "dq", "tau")

LOG_DQ_IS_COMMANDED = Constant(
    "LOG_DQ_IS_COMMANDED", True, "bool", Provenance.MEASURED,
    "300Hz 日志的 `dq` 是**指令** dq（`dq_s`），**不是实测**。"
    "⚠ 且出厂 `ff_mask=0x1BF` 含 `FF_VELREF` ⇒ 固件把 `kd·dq` 移进 tau 后**把 `dq_s` 清零** "
    "⇒ **出厂配置下日志的 `dq` 列恒为 0**。要拿真值须临时清 VELREF（0x27），跑完恢复。")

LOG_HAS_MEASURED_Q = Constant(
    "LOG_HAS_MEASURED_Q", False, "bool", Provenance.MEASURED,
    "❌ 日志里**没有实测位置**。实测 q/dq/tau/温度只在 100Hz 状态帧里。"
    "⇒ 一切「参考 vs 实测」的指标都必须**跨流对齐**（见 align.py），"
    "而跨流对齐有它自己的失效模式（时间原点、δ 顶窗边）。")

LOG_HAS_TEMPERATURE = Constant("LOG_HAS_TEMPERATURE", False, "bool", Provenance.MEASURED,
                              "❌ 日志无温度。温度只能从 100Hz 状态帧按 10Hz 有效速率取。")

LOG_HAS_FLAGS = Constant("LOG_HAS_FLAGS", False, "bool", Provenance.MEASURED,
                         "❌ 日志无 flags/err/joint_fault ⇒ 日志段内发生的锁存**看不到**，"
                         "必须同时订阅状态帧的 flags。")


# --------------------------------------------------------------------------- #
# 持位段语义（2026-09-20 持位交接修复前后**含义不同**）
# --------------------------------------------------------------------------- #

HOLD_QR_IS_ANCHOR = Constant(
    "HOLD_QR_IS_ANCHOR", True, "bool", Provenance.MEASURED,
    "进持位后 `cmd->q_ref` 被**冻结**成锚点，此后每拍被覆写（`control_loop.c` 的 ht 块）。"
    "⇒ 持位段 `q_ref` ≠ 实测，两者的差 = 静差（摩擦拖曳 + 模型误差）。")

HOLD_ANCHOR_AFTER_FIX = Constant(
    "HOLD_ANCHOR_AFTER_FIX", "参考位置（clamp 后）", "str", Provenance.MEASURED,
    "固件 ≥ `feat/hold-handoff`（2026-09-20）：锚点 = 交接那一拍的 `clamp(cmd->q_ref)`。"
    "**修复前**是「交接那一刻的实测位置」—— 那时静态跟踪误差被**冻结成永久停位偏差**"
    "（真机 J2 −12.6 mrad）。⇒ 分析历史数据时必须先确认固件版本，否则「停位偏差」"
    "这一个量在两代固件上**物理含义不同**。")


# --------------------------------------------------------------------------- #
# 位置越限包络 —— 判「还能不能安全动臂」**必须量到这条线**，不是软限
# --------------------------------------------------------------------------- #

POS_LATCH_BAND = Constant(
    "POS_LATCH_BAND", 0.05, "rad", Provenance.MEASURED,
    "固件位置越限的**判定余量**：锁存发生在 `q > q_max + 本值` 或 `q < q_min − 本值`，"
    "**不是**在软限本身（`safety_check.c:165`：`st->q < jp->q_min - 0.05f || "
    "st->q > jp->q_max + 0.05f`）。"
    "⇒ **任何「离危险还有多远」的量都必须量到 `软限 ± POS_LATCH_BAND`**。"
    "量到软限会系统性少算 0.05 —— 对 J4（软限上界仅 **+0.0175**，见下）会得出"
    "**恒不可达**的判据，即「任何运行都被拒绝」（2026-09-20 真机首次暴露）。")

POS_RECOVER_BAND = Constant(
    "POS_RECOVER_BAND", 0.05, "rad", Provenance.MEASURED,
    "固件 `safety_check.c:94`。**未使能 + 越界幅度 ≤ 本值 ⇒ 不锁存**，留给 `enable` 时的"
    "受控回程把轴带回（`safety_check.c:183` 的 J4 陷阱修复）。"
    "⇒ 失能的臂小幅漂出软限**不是**故障；但**使能态**下同样的越界照旧锁存。")

POS_DEBOUNCE_CYCLES = Constant(
    "POS_DEBOUNCE_CYCLES", 5, "拍", Provenance.MEASURED,
    "固件 `safety_check.c:84`。越限要**连续 5 拍**（@300Hz ≈ 17ms）才置 `joint_fault`；"
    "另有 `FB_STABLE_FRAMES = 8` 的开机免判期。"
    "⇒ 单拍野值不锁存，但**锁存一旦置位不受 enabled 门控、`reset` 也清不掉**。")

#: J4 的软限上界 —— **全轴最紧**，且是**单侧**的（行程 −3.0715…+0.0175，零位天然距上界
#: 仅 ~0.0097 rad）。这是"对称裕度判据必然失败"的根源，故单列出来当反例锚。
J4_QMAX_RAD = Constant(
    "J4_QMAX_RAD", 0.017547, "rad", Provenance.MEASURED,
    "固件 `joint_limit_macros.h:16` 的 `JL_QMAX_J4`（= +1°）。**J4 零位天然就在它下面**"
    "—— 2026-09-20 真机零位实测 `q4 = +0.0078`，距软限仅 **0.0097**、距锁存线 **0.0597**。"
    "⇒ 拿软限当基准的任何固定裕度判据都会把 J4 的**正常零位**判成危险。")


def latch_margin_rad(q: float, q_min: float, q_max: float) -> float:
    """`q` 距**锁存线**的裕度（rad）：``min(q − (q_min − B), (q_max + B) − q)``，B = 判定余量。

    ⚠ **这是唯一该用来判"还能不能动"的裕度**。它回答的是"离固件真的置 `joint_fault` 还有
    多远" —— 而那才是会让整臂锁死、且 `reset` 清不掉的那条线。

    ⚠ 别改成量到软限：软限只是**指令钳位**的边界（固件把 `q_ref` 钳在限内），
    臂可以正常顶在软限上工作。量到软限会少算 `POS_LATCH_BAND`。
    """
    b = float(POS_LATCH_BAND.value)
    return min(float(q) - (float(q_min) - b), (float(q_max) + b) - float(q))


def inside_soft_limits(q: float, q_min: float, q_max: float) -> bool:
    """是否在**软限内**。越限本身不是故障（指令会被钳回），但**使能态**下越限会走向锁存。"""
    return float(q_min) <= float(q) <= float(q_max)


# --------------------------------------------------------------------------- #
# 取不到的量 —— 显式登记，别让指标静默返回 0
# --------------------------------------------------------------------------- #

UNAVAILABLE: Dict[str, str] = {
    "bus_voltage": "状态帧与电机反馈结构体都无电压字段（`dm_motor_drv.h` 的 fbpara 只有 "
                   "state/pos/vel/tor/Kp/Kd/Tmos/Tcoil）⇒ **取不到**。",
    "bus_current": "同上，无任何电流字段 ⇒ **取不到**。",
    "phase_current": "同上 ⇒ 取不到。",
    "fk_arbitrary_q": "固件没有「给 q 求位姿」的下行命令；`get_tcp` 只能算当前反馈 q "
                      "⇒ **任意构型的 FK 取不到**（PC 侧要算须自己带运动学模型）。",
    "drop_hold": "状态帧无该位；SDK 只有**推断**属性与 `ERR{cmd,0x06}` 的确证路径 ⇒ 读不到确定值。",
    "loop_jitter_per_tick": "只有累积计数（`loop_k` 峰值千 cycle / `ovr` 超预算拍数）"
                            "⇒ **逐拍控制环时序取不到**。",
    "motor_err_decode": "电机 `err` 字节 SDK 无解码表，只有 raw int。",
    "cart_progress": "笛卡尔 `0x4E` 只在**规划结束后**发一次（n_wp/plan_us）"
                     "⇒ 在途进度 / 剩余时间 / dry-run 取不到。",
}


# --------------------------------------------------------------------------- #
# 派生量
# --------------------------------------------------------------------------- #

def kp_hold(mit_kp: float, hold_kp_gain: float, cap: float = 500.0) -> float:
    """持位刚度 = `min(mit_kp × hold_kp_gain, 500)`。

    ⚠ 500 是**固件里的钳位**（`control_loop.c` 的 ht 块），不是可调参数。
    ⚠ `hold_kp_gain` 出厂 2.0 ⇒ **J1~J4 全部被钳到 500**（400×2=800、300×2=600 都超），
    只有 J5~J7 真的是 2×（50→100）。**把「×2」当通式会把 J1~J4 的持位刚度算错 1.6~2×。**
    """
    return min(mit_kp * hold_kp_gain, cap)


def static_error_floor_rad(fc0: float, fc1: float, kp_hold_val: float) -> float:
    """持位态**静差地板**（rad）：`max(|fc0|,|fc1|) / kp_hold`。

    前提：持位态内置前馈（积分/摩擦）**全停** ⇒ 稳态只由 `G(q_hold)` 与 `kp_hold·e` 平衡，
    库仑摩擦最多把臂拖开 `fc0/kp_hold`。⇒ 这是**物理下界**，静差不可能低于它。

    ⚠ 前提失效的场合：`hold`（看门狗/掉线）与 ZERO_G 下走的是别的分支 ⇒ 别用它判那些态。
    ⚠ `kp_hold <= 0` 时刚度为零 ⇒ 地板**无界**，返回 ``inf``（调用侧必须当"跳过"处理，
    **不能**当 0 —— 把无界当成 0 会让"静差/地板"这个比值恒为 0 分，那是最坏的一种假信号）。
    """
    if not (kp_hold_val > 0):
        return float("inf")
    return max(abs(fc0), abs(fc1)) / kp_hold_val


ALL_CONSTANTS: List[Constant] = [
    CTRL_HZ, STATUS_HZ, CTRL_DT, LOG_CAP_TICKS, LOG_CAP_S, LOG_CHUNK_BYTES,
    Q_LSB_RAD, DQ_TRUNCATION_LSB, LOG_DQ_IS_COMMANDED, LOG_HAS_MEASURED_Q,
    LOG_HAS_TEMPERATURE, LOG_HAS_FLAGS, HOLD_QR_IS_ANCHOR, HOLD_ANCHOR_AFTER_FIX,
    POS_LATCH_BAND, POS_RECOVER_BAND, POS_DEBOUNCE_CYCLES, J4_QMAX_RAD,
    *TAU_LSB_NM.values(), *DQ_LSB_RAD_S.values(),
]


def describe(only_uncertain: bool = False) -> str:
    """给报告用的仪表边界清单。`only_uncertain=True` 只列推导/占位项。"""
    lines = ["| 常数 | 值 | 单位 | 证据 | 说明 |", "|---|---:|---|---|---|"]
    for c in ALL_CONSTANTS:
        if only_uncertain and c.prov in (Provenance.MEASURED, Provenance.NOMINAL):
            continue
        lines.append(f"| `{c.key}` | {c.value} | {c.unit} | {c.prov.value} | {c.note} |")
    if UNAVAILABLE:
        lines += ["", "**取不到的量（指标会标「未实测」，不编分）**：", ""]
        for k, why in UNAVAILABLE.items():
            lines.append(f"- `{k}`：{why}")
    return "\n".join(lines)


def tau_lsb(n_joints: int) -> Optional[float]:
    """本机最常见的 LSB（全轴取最大）—— 只有需要单一地板时用；逐轴分析请查表。"""
    vals = [TAU_LSB_NM[i].value for i in range(1, n_joints + 1) if i in TAU_LSB_NM]
    return max(vals) if vals else None
