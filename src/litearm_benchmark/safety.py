"""安全前置与不变量 —— 每一次动臂之前必须过这道闸。

这个模块的存在理由是几条**真实的、踩过的**事故（都记在项目知识库里）：

* **J4 软限锁死**：位置越限锁存**不受 enabled 门控**，`ERR 0x06` 且 `reset` 清不掉；
  停机前必须把 J4 停在限内。J4 距上软限只有 **0.0175 rad**，是所有轴里最危险的。
* **失能会漂出软限位**：位置环一断软限位形同虚设；任何轴越限 ⇒ 整个笛卡尔面失效。
* **`zero_g_stop` 是异步的**：只发帧不等确认，之后仍有约 677ms 固件拒运动命令。
* **破坏性脚本开头必须断言起点干净**。
* **绝不 `disable()`**（见下 `LEAVE_ENABLED` 的说明）。

⚠ 本模块**只做检查与拒绝**，不替调用方决定"要不要继续"。拒绝时抛 :class:`SafetyAbort`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

#: 任何轴距软限的**最小可接受裕度**。低于它直接拒绝动臂。
#: 依据：J4 的软限上界仅 +0.0175 rad，而位置包络的判定余量是 0.05 rad；留 0.05 是
#: 「比包络余量还宽一档」的经验值 —— 裕度低于它时，一次正常的跟踪超调就可能触到包络。
DEFAULT_MIN_MARGIN_RAD = 0.05

#: `disable()` 会掉力矩（扶着的人吃力）并触发 `drop_hold` 锁存 ⇒ **基准测试绝不主动失能**。
#: 收尾统一走 `park()`（mode=0）+ 回零位保持，而不是失能。
LEAVE_ENABLED = True


class SafetyAbort(RuntimeError):
    """安全前置未通过 —— 调用方**不应**catch 后继续动臂。"""


@dataclass
class PreflightReport:
    ok: bool
    margins: List[float] = field(default_factory=list)      # 逐轴 min(下余, 上余)
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
        return (f"最小裕度 {self.min_margin:+.4f} rad @J{self.tightest}"
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


def preflight(arm, *, min_margin_rad: float = DEFAULT_MIN_MARGIN_RAD,
              limits: Optional[Sequence[tuple]] = None,
              allow_faults: bool = False) -> PreflightReport:
    """动臂前的安全前置。**只读**，不发任何运动命令。

    检查项：
      1. 能拿到位置与软限位；
      2. 逐轴裕度 ≥ ``min_margin_rad``；
      3. `joint_fault == 0`（锁存轴会让整臂静默降级，且"回落"不可观测）；
      4. `drop_hold` 推断为真时告警（只能推断，状态帧没有该位）。
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
    rep.margins = [min(q[i] - lim[i][0], lim[i][1] - q[i]) for i in range(len(q))]

    if st.joint_fault and not allow_faults:
        axes = [i + 1 for i in range(len(q)) if st.joint_fault & (1 << i)]
        rep.faults.append(f"joint_fault=0x{st.joint_fault:04X} 锁存轴 {axes}")
    if getattr(st, "drop_hold_inferred", False):
        rep.warnings.append("drop_hold 推断为真（状态帧无该位，只能推断）")
    if rep.min_margin < min_margin_rad:
        rep.faults.append(
            f"J{rep.tightest} 裕度 {rep.min_margin:+.4f} rad < 要求 {min_margin_rad:.3f} "
            f"—— 越限锁存**不受 enabled 门控**且 reset 清不掉，拒绝动臂")

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
