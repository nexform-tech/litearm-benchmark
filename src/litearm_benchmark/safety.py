"""安全前置与不变量 —— 每一次动臂之前必须过这道闸。

这个模块的存在理由是几条**真实的、踩过的**事故（都记在项目知识库里）：

* **J4 软限锁死**：位置越限锁存**不受 enabled 门控**，`ERR 0x06` 且 `reset` 清不掉。
  ⚠ 锁存线是 `软限 ± POS_LATCH_BAND`（固件 `safety_check.c:165`），**不是软限本身** ——
  本模块 2026-09-20 首次上真机时正是因为量错了这条线，把 J4（软限上界仅 +0.0175、
  零位天然距它 0.0097）的**正常零位**判成危险，导致整档跑不起来。详见
  :func:`litearm_benchmark.instruments.latch_margin_rad`。
* **失能会漂出软限位**：位置环一断软限位形同虚设；任何轴越限 ⇒ 整个笛卡尔面失效。
* **`zero_g_stop` 是异步的**：只发帧不等确认，之后仍有约 677ms 固件拒运动命令。
* **破坏性脚本开头必须断言起点干净**。
* **绝不 `disable()`**（见下 `LEAVE_ENABLED` 的说明）。

⚠ 本模块**只做检查与拒绝**，不替调用方决定"要不要继续"。拒绝时抛 :class:`SafetyAbort`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from . import instruments as I

#: **动作闸**（`Executor._gate`）对**目标位形**距**锁存线**（`软限 ± POS_LATCH_BAND`）
#: 的最小裕度要求。
#:
#: ⚠⚠ **量到锁存线，不是量到软限** —— 这一条是 2026-09-20 真机首次上机时暴露的：
#: 原实现（与本文档的早期版本）量到软限，于是 J4 —— 软限上界仅 **+0.0175**、零位天然
#: 距它只有 **0.0097** —— 的**正常零位**被永久判成危险，三道判据（0.05/0.02/0.01）
#: 全部恒不可达 ⇒ **benchmark 在这台臂上一次也跑不起来**。
#: 固件的锁存判定在 `safety_check.c:165` 的 `q_max + 0.05`，软限只是**指令钳位**边界，
#: 顶在软限上是正常工作状态（见 :func:`litearm_benchmark.instruments.latch_margin_rad`）。
#:
#: ⚠ 另注：**在软限内的点距锁存线恒 > `POS_LATCH_BAND`** ⇒ 用它当"起点裕度"判据是空转；
#: 它真正把关的地方是**目标位形**（`Executor._gate`），那里才需要"离限位还有点余量"。
MIN_TARGET_MARGIN_RAD = 0.02

#: `disable()` 会掉力矩（扶着的人吃力）并触发 `drop_hold` 锁存 ⇒ **基准测试绝不主动失能**。
#: 收尾统一走 `park()`（mode=0）+ 回零位保持，而不是失能。
LEAVE_ENABLED = True


class SafetyAbort(RuntimeError):
    """安全前置未通过 —— 调用方**不应**catch 后继续动臂。"""


@dataclass
class PreflightReport:
    ok: bool
    margins: List[float] = field(default_factory=list)      # 逐轴**距锁存线**裕度
    q: List[float] = field(default_factory=list)
    faults: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def min_margin(self) -> float:
        return min(self.margins) if self.margins else float("nan")

    @property
    def tightest(self) -> int:
        """裕度最小的轴号（1-based）；无数据时 0。"""
        return (self.margins.index(self.min_margin) + 1) if self.margins else 0

    def render(self) -> str:
        if not self.margins:
            return "（无位置数据）"
        return (f"最小距锁存线裕度 {self.min_margin:+.4f} rad @J{self.tightest}"
                f"（q={self.q[self.tightest - 1]:+.4f}）")


def _limits(arm) -> List[tuple]:
    """逐轴 `(q_min, q_max)`。**先问固件(0x24)，拿不到才退回出厂表**并记警告。

    ⚠ 0x24 读回的是**设备当天值**，而 `defaults.c` 是出厂值 —— 两者可能不同
    （本项目实测过 fc0 差 1.5×）。安全判断必须用设备值。
    """
    try:
        ps = arm.params.all_joint_params()
        if ps:
            return [(p.q_min, p.q_max) for p in ps]
    except Exception:                                      # noqa: BLE001
        pass
    raise SafetyAbort("读不到关节软限位（0x24）—— 安全闸不能靠猜，拒绝动臂。"
                      "请确认已连接并使能，或显式传入 limits。")


def preflight(arm, *, limits: Optional[Sequence[tuple]] = None,
              allow_faults: bool = False) -> PreflightReport:
    """动臂前的安全前置。**只读**，不发任何运动命令。

    检查项（**判据一律照抄固件的锁存语义**，见 `safety_check.c:165-190`）：
      1. 能拿到位置与软限位；
      2. `joint_fault == 0`（锁存轴会让整臂静默降级，且"回落"不可观测）；
      3. 没有轴**越过锁存线**（`软限 ± POS_LATCH_BAND`）—— 那才是会锁死的那条线；
         唯一的例外是固件自己认的**失能 + 幅度 ≤ `POS_RECOVER_BAND`**（可受控回程）；
      4. `drop_hold` 推断为真时告警（只能推断，状态帧没有该位）。

    ⚠ **为什么没有"距软限 ≥ X"这类通用裕度判据**：固件的锁存判定点是**包络**
    （`q_max + 0.05`），软限只是指令钳位边界。于是任何**在软限内**的点距锁存线**恒 > 0.05**
    ⇒ 这类判据对限内是空转；而它在限外又会把"失能漂出软限"这个**常规可恢复态**
    （固件专门为它写了回程宽限）误杀成故障。2026-09-20 真机首跑就是被它挡死的。
    """
    st_msg = arm.get_state(refresh=True)
    st = st_msg.value
    if st is None:
        raise SafetyAbort("读不到状态帧 —— 拒绝动臂。")

    lim = list(limits) if limits is not None else _limits(arm)
    q = [j.q for j in st.joints]
    if len(q) != len(lim):
        raise SafetyAbort(f"轴数不匹配：状态帧 {len(q)} vs 限位 {len(lim)}")

    rep = PreflightReport(ok=True, q=q)
    rep.margins = [I.latch_margin_rad(q[i], lim[i][0], lim[i][1]) for i in range(len(q))]

    if st.joint_fault and not allow_faults:
        axes = [i + 1 for i in range(len(q)) if st.joint_fault & (1 << i)]
        rep.faults.append(f"joint_fault=0x{st.joint_fault:04X} 锁存轴 {axes}")
    if getattr(st, "drop_hold_inferred", False):
        rep.warnings.append("drop_hold 推断为真（状态帧无该位，只能推断）")

    # ① 越过锁存线 —— 固件只在**这个区域**里才可能置 `joint_fault`（`safety_check.c:165`
    #    的判定点就在包络上）。唯一的豁免是它自己的失能回程宽限。
    beyond = [i for i in range(len(q)) if rep.margins[i] <= 0.0]
    if beyond:
        axes = [i + 1 for i in beyond]
        worst = min(rep.margins[i] for i in beyond)          # 越线最多（负得最多）
        enabled = bool(getattr(st, "enabled", False))
        if not enabled and -worst <= I.POS_RECOVER_BAND.value:
            rep.warnings.append(
                f"轴 {axes} 越出锁存线 {(-worst):.4f} rad，但**失能态且幅度在 "
                f"`POS_RECOVER_BAND` {I.POS_RECOVER_BAND.value} 内** —— 固件不锁存，"
                f"`enable` 的受控回程会把它带回（`safety_check.c:183`）；继续")
        else:
            why = ("当前**已使能**，回程宽限不适用" if enabled
                   else f"越线幅度 {-worst:.4f} > `POS_RECOVER_BAND`")
            rep.faults.append(
                f"轴 {axes} 越过**锁存线** {(-worst):.4f} rad 且{why}"
                f"—— 固件连续 {I.POS_DEBOUNCE_CYCLES.value} 拍后置 `joint_fault`，"
                f"而它**不受 enabled 门控、`reset` 也清不掉** ⇒ 拒绝动臂")

    # ② 软限外但**未越锁存线** —— 固件在这个区间**根本不进锁存分支**（两态都不锁存），
    #    属常规态（失能的臂会漂出去）。只告警，并提示第一个动作应把它带回限内。
    outside = [i + 1 for i in range(len(q))
               if rep.margins[i] > 0.0
               and not I.inside_soft_limits(q[i], lim[i][0], lim[i][1])]
    if outside:
        rep.warnings.append(
            f"轴 {outside} 在**软限外但未越锁存线** —— 固件不会锁存（判定点在包络上），"
            f"可直接跑；但应尽快用动作把它带回限内（越限时 `q_ref` 会被钳）")

    rep.ok = not rep.faults
    return rep


def require(arm, **kw) -> PreflightReport:
    """:func:`preflight` 的抛异常版本 —— 基准测试的主路径应当用这个。"""
    rep = preflight(arm, **kw)
    if not rep.ok:
        raise SafetyAbort("安全前置未通过：" + "；".join(rep.faults) + f"（{rep.render()}）")
    return rep


def park(arm, *, zero: bool = True, timeout: float = 30.0) -> None:
    """收尾：回零位并**保持使能**（绝不 `disable`）。失败只告警，不再抛。

    为什么不失能：`disable` 会掉力矩并触发 `drop_hold` 锁存，而锁存解除要走
    `reset→enable`；基准测试的收尾不该给下一次运行埋这个坑。
    """
    try:
        if zero:
            arm.movej([0.0] * arm.n, speed=0.3)
        arm.park()
    except Exception as exc:                               # noqa: BLE001
        import warnings
        warnings.warn(f"收尾回零/停靠失败（臂仍在保持，未失能）：{exc!r}", stacklevel=2)
