"""套件骨架 —— **动作是数据，不是代码**。

档位（快/标准/完整/需人工）之间的差别只是 :data:`Action` 的列表不同，
执行器只有一份。这样做的好处：想加一个指标时，只需要往对应档位里加一条动作，
不用碰任何执行逻辑；也让"这一档到底跑了什么"变成可以打印、可以 diff 的东西。

⚠ 动作一律在**安全台架包线**内（见 :data:`POSES`）：不碰 J4 的软限上界、不进奇异、
不贴限位。需要大范围/负载的动作一律进 `manual` 档，且**必须**标 :data:`MANUAL`。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .. import instruments as I
from ..metrics.analysis import MIN_EXCITATION_RAD, Segment

#: 需人工介入的动作标记 —— 无人值守档位里出现就要拒绝运行。
MANUAL = "manual"

#: 安全台架包线里的位形。**都避开了 J4 的软限上界（仅 +0.0175 rad）与 URDF 零位奇异。**
POSES: Dict[str, List[float]] = {
    #: URDF 零位。⚠ 它是一个**奇异位形**（雅可比 σmin≈0）⇒ 只当收尾/起点，不当笛卡尔端点。
    "zero": [0.0] * 7,
    #: 舒展位形（工具链里叫 TEST_Q0；σmin 在候选里最大）。
    "bench": [0.0, -0.60, 0.90, -1.20, 0.0, 0.50, 0.0],
    #: 舒展位形附近的小偏离 —— 做重复定位/静差用。
    "bench_j2lo": [0.0, -1.10, 0.90, -1.20, 0.0, 0.50, 0.0],
    "bench_j2hi": [0.0, -0.30, 0.90, -1.20, 0.0, 0.50, 0.0],
    #: 多轴联动（J1 与 J3 同时动）—— 给同步与跟踪用。
    "bench_multi": [0.7, -0.85, 1.30, -1.20, 0.30, 0.50, 0.20],
    "bench_multi2": [-0.7, -0.85, 1.30, -1.20, -0.30, 0.50, -0.20],
}


@dataclass
class Action:
    """一条动作。``kind`` 决定执行器怎么解释其余字段。"""

    kind: str
    label: str = ""
    target: Optional[str] = None       # POSES 的键名（关节类动作）
    #: 每轮的**起点**位形。给了它，`reps` 的每一轮都先走到 `via`、再落到 `target`。
    #:
    #: ⚠ **没有它，`reps` 是假的**：第 2..N 轮的目标与上一轮落点**完全一样**，臂一步都不动，
    #: 那一轮的"落点"只是上一轮的复制 —— 重复性变成测传感器噪声、跟踪段变成空动作。
    #: 2026-09-20 真机首跑实测：`小位移@0.3` 三次重复的段长全是 0.02 s（`gcheck` 恰好
    #: 已把臂停在 `bench_j2lo`），而 `static.repeat_ptp_mrad` 拿着那 0.76 mrad 判了 91 分。
    via: Optional[str] = None
    speed: float = 0.4
    n_ticks: int = 900                 # 300Hz 日志窗口拍数（≤2400）
    reps: int = 1
    rest_s: float = 1.0
    wait_s: float = 0.0
    manual_hint: str = ""              # kind == "manual" 时提示操作者做什么
    requires_manual: bool = False
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def is_manual(self) -> bool:
        return self.kind == "manual" or self.requires_manual


# --------------------------------------------------------------------------- #
class Executor:
    """把 :class:`Action` 列表跑成 :class:`Segment` 列表 + 采集数据。"""

    def __init__(self, session, capture, *, interactive: bool = True,
                 log: Optional[Callable[[str], None]] = None):
        self.s = session
        self.cap = capture
        self.arm = session.arm
        self.trace = capture.trace
        self.interactive = interactive
        self._log = log or (lambda m: None)
        self.segments: List[Segment] = []
        self.reconnects = 0

    # -- 基础设施 ---------------------------------------------------------- #
    def _pose(self, key: str) -> List[float]:
        if key not in POSES:
            raise KeyError(f"未知位形 {key!r}；可选 {sorted(POSES)}")
        return list(POSES[key])

    def _gate(self, target: Sequence[float], label: str) -> None:
        """每一次动臂前的**限位闸** —— 裕度不足直接抛，不给"就这一次"留口子。

        ⚠ 裕度一律取**距锁存线**（`软限 ± POS_LATCH_BAND`），不取距软限 ——
        见 :func:`litearm_benchmark.instruments.latch_margin_rad`。
        另外**目标必须在软限内**：越限时固件会把 `q_ref` 钳回，动作等于白做。
        """
        from ..safety import MIN_TARGET_MARGIN_RAD, SafetyAbort
        from ..instruments import latch_margin_rad, inside_soft_limits
        n = len(target)
        lim = self.s.limits

        bad = [i + 1 for i in range(n) if not inside_soft_limits(target[i], lim[i][0], lim[i][1])]
        if bad:
            raise SafetyAbort(f"动作 {label!r} 的目标在**软限外**（轴 {bad}）⇒ 拒绝执行"
                              f"（固件会把 q_ref 钳回限内，这一动作不会真的到位）")

        worst = min(latch_margin_rad(target[i], lim[i][0], lim[i][1]) for i in range(n))
        if worst < MIN_TARGET_MARGIN_RAD:
            raise SafetyAbort(f"动作 {label!r} 的目标位形距锁存线只有 {worst:+.4f} rad ⇒ 拒绝执行")

        cur = self.trace.q[-1] if self.trace.q.size else np.asarray(target)
        cur_margin = min(latch_margin_rad(float(cur[i]), lim[i][0], lim[i][1])
                         for i in range(min(len(cur), n)))
        if cur_margin < 0.01:
            raise SafetyAbort(f"动作 {label!r} 的**起点**距锁存线只有 {cur_margin:+.4f} rad ⇒ 拒绝执行")

    def _at_rest(self, timeout: float = 12.0) -> bool:
        """到静止：**用位置判，不用 dq** —— 上报 dq 的量化 LSB 是 0.00977 rad/s，
        阈值写在 1e-3 是**低于物理地板**的判据（本项目真踩过，恒不可达）。"""
        t0 = time.monotonic()
        q1 = self._q()
        while time.monotonic() - t0 < timeout:
            time.sleep(0.3)
            q2 = self._q()
            if np.abs(q2 - q1).max() < 1e-3:
                return True
            q1 = q2
        return False

    #: 笛卡尔停稳判据的三个数。**两个条件缺一不可**：
    #: 「已接近目标」且「窗口内不再动」。
    #: ⚠ 只看"没在动"会把**规划/启动之前**那一瞬判成停稳 —— 那一刻臂还没开始走，
    #:   当然不动。真机实测：`wait=False` 下 core 不会自己回读 TCP 做到位判定，
    #:   所以"停没停"必须由调用方自己判。
    TCP_SETTLE_NEAR_M = 0.006      # 与 core `move_l` 的到位判据一致（6mm）
    TCP_SETTLE_BAND_M = 0.0003     # 0.3mm
    TCP_SETTLE_WINDOW_S = 0.4

    def _tcp_settled(self, tt, pp, goal) -> bool:
        """TCP 是否**停在了目标附近**。`tt`/`pp` 是逐步累积的采样。"""
        if len(pp) < 3:
            return False
        if float(np.linalg.norm(np.asarray(pp[-1], float) - goal)) > self.TCP_SETTLE_NEAR_M:
            return False
        t_end = tt[-1]
        j = len(tt) - 1
        while j > 0 and t_end - tt[j] < self.TCP_SETTLE_WINDOW_S:
            j -= 1
        if t_end - tt[j] < self.TCP_SETTLE_WINDOW_S * 0.8:   # 窗口还没铺满
            return False
        span = float(np.linalg.norm(np.asarray(pp[-1], float) - np.asarray(pp[j], float)))
        return span < self.TCP_SETTLE_BAND_M

    def _q(self) -> np.ndarray:
        st = self.arm.get_state(refresh=True).value
        return np.asarray([j.q for j in st.joints], float)

    def _hold(self, seconds: float) -> None:
        """保持 `seconds` 秒，**期间主动采样** —— 绝不用裸 `time.sleep`。

        ⚠ 为什么必须主动读：状态帧只在**有人读串口**时才被 `_on_status` 收上来
        （core 没有读线程）。裸 `sleep` 期间一帧都收不到 ⇒ 静止段的窗口**零样本**，
        `_rest_hold_windows` 会把整段丢掉，或者更糟：只剩零星几帧、静差算在一个
        跟本次动作无关的时刻上。

        ⚠ 这也正是 2026-09-20 真机首跑"静差 798 mrad"的两个成因之一（另一个是
        `capture._flush` 压平时间轴）。
        """
        end = time.monotonic() + max(0.0, float(seconds))
        while True:
            self.arm.get_state(refresh=True)     # 读 ⇒ 帧进 trace（超时返回 None，不抛）
            if time.monotonic() >= end:
                return
            time.sleep(0.01)

    def _mark_rest(self, label: str, target: Optional[Sequence[float]],
                   *, travel_rad: Optional[float] = None) -> None:
        """标一个静止段。**两端都取 `time.monotonic()`** —— 与 `Segment` 的时间源一致。

        ⚠ 旧实现把 `t0` 取成 `trace.t_status[-1]`（**上一帧的时刻**，而静止期本来就没帧
        ⇒ 它是"上一个动作结束时"的旧快照），`t1` 却取 `time.monotonic()` ⇒ 段自身
        时长算成 7.8~11.2 s（真实只有 0.2 s），且与 `t_status` 不同源。

        `travel_rad` 从**刚结束的那个动作**继承 —— 重复性指标要靠它判"这一次落点到底是
        走过来的、还是原地没动"。
        """
        t0 = time.monotonic()
        self._hold(0.2)
        self.segments.append(Segment(label=label, kind="rest", t0=t0,
                                     t1=time.monotonic(),
                                     target=list(target) if target is not None else None,
                                     travel_rad=travel_rad))

    # -- 动作执行器 -------------------------------------------------------- #
    def run(self, actions: Sequence[Action], *, deadline: Optional[float] = None) -> List[Segment]:
        for a in actions:
            if deadline and time.monotonic() > deadline:
                self._log(f"⚠ 到时预算用尽，跑完剩余动作被迫截断（已跑 {len(self.segments)} 段）")
                break
            self._one(a)
        return self.segments

    def _one(self, a: Action) -> None:
        if a.is_manual:
            self._manual(a)
            return
        kind = a.kind
        if kind == "rest":
            self._mark_rest(a.label or "rest", None)
            return
        if kind == "wait":
            self._hold(a.wait_s or 1.0)
            return
        if kind in ("movej", "movej_sync"):
            self._joint(a)
            return
        if kind in ("cart_line", "cart_arc"):
            self._cart(a)
            return
        if kind == "watchdog":
            self._watchdog(a)
            return
        if kind == "kin_bench":
            self._kin_bench(a)
            return
        if kind == "gcheck":
            self._gcheck(a)
            return
        if kind == "drift_ref":
            self._drift_ref(a)
            return
        raise ValueError(f"未知动作类型 {kind!r}")

    def _joint(self, a: Action) -> None:
        tgt = self._pose(a.target or "bench")
        via = self._pose(a.via) if a.via else None
        if via is not None:
            need = float(np.max(np.abs(np.asarray(tgt) - np.asarray(via))))
            if need < MIN_EXCITATION_RAD:
                raise ValueError(
                    f"动作 {a.label!r} 的 via({a.via}) 与 target({a.target}) 只差 "
                    f"{need * 1000:.1f} mrad < {MIN_EXCITATION_RAD * 1000:.0f} mrad "
                    f"⇒ 这个「重复」不会有位移，位形表自己退化了")
        for rep in range(a.reps):
            label = f"{a.label or a.kind}#{rep}" if a.reps > 1 else (a.label or a.kind)
            if via is not None:
                # 先把臂挪回本轮**起点** —— 不记 300Hz 窗口（这一段不参与分析），
                # 但仍要过限位闸，且它会在 100Hz 时间轴上留下痕迹。
                self._gate(via, label + ":via")
                self.arm.movej(via, speed=a.speed)
            q_before = self._q()          # 实测起点 ⇒ travel 是**量出来的**，不是照目标算的
            self._gate(tgt, label)
            with self.cap.window(a.n_ticks, label) as w:
                t0 = time.monotonic()
                seg = Segment(label=label, kind=a.kind, t0=t0, t1=t0,
                              target=list(tgt),
                              # 窗口退出时才 append ⇒ 此刻的 len 就是它将要拿到的下标
                              chunk=len(self.trace.chunks),
                              extra={"speed": a.speed})
                self.trace.mark(label + ":t0")
                fn = self.arm.movej_sync if a.kind == "movej_sync" else self.arm.movej
                fn(tgt, speed=a.speed)
                seg.t1 = time.monotonic()
                seg.travel_rad = float(np.max(np.abs(self._q() - q_before)))
                self.segments.append(seg)
            trav = (f"{seg.travel_rad * 1000:6.1f} mrad"
                    if seg.travel_rad is not None else "  ——  ")
            self._log(f"  {label}: {'sync' if a.kind=='movej_sync' else 'movej'} "
                      f"@speed {a.speed} 行程 {trav} → {np.round(tgt, 3).tolist()}")
            if a.rest_s:
                self._hold(a.rest_s)
                self._mark_rest(label + ":rest", tgt, travel_rad=seg.travel_rad)

    def _cart(self, a: Action) -> None:
        """笛卡尔：``wait=False`` + **同线程**轮询 `get_tcp`/`poll_cart`。

        ⚠ 绝不开第二个线程去读 TCP —— core 没有读线程，"等应答与读帧是同一件事"，
        第二个读口会分吃字节流（本项目明确记载过这一点）。
        """
        tcp = self.arm.get_tcp().value
        if tcp is None:
            self._log("  ⚠ 读不到 TCP ⇒ 跳过笛卡尔段")
            return
        p0 = list(tcp[:3]); rpy = list(tcp[3:])
        if a.kind == "cart_line":
            goal = [p0[0], p0[1] + float(a.extra.get("dy", 0.05)), p0[2]] + rpy
            poses = [goal]
        else:
            u = np.array([p0[0], p0[1], 0.0])
            u = u / max(np.linalg.norm(u), 1e-9)
            c = np.asarray(p0) - 0.04 * u
            v = np.cross([0.0, 0.0, 1.0], u)
            via = c + 0.04 * (np.cos(np.pi / 4) * u + np.sin(np.pi / 4) * v)
            goal = list(c + 0.04 * v) + rpy
            # ⚠ **`pose_start` 必须是 `p0` 本身，不是圆心 `c`** —— `move_c` 收的是
            #   **三点定圆**（起点 + via + 终点），且固件会把 `pose_start` 与**当前实测 TCP**
            #   比对（容差 6mm / 0.03rad）。给圆心会直接抛 `InvalidCommandError`：
            #   圆心离实测 TCP 恒差一个半径（这里 40mm）。
            poses = [list(p0) + rpy, list(via) + rpy, goal]
        label = a.label or a.kind
        with self.cap.window(a.n_ticks, label):
            t0 = time.monotonic()
            try:
                if a.kind == "cart_line":
                    plan = self.arm.move_l(tuple(poses[0]), speed=a.speed, wait=False)
                else:
                    plan = self.arm.move_c(tuple(poses[0]), tuple(poses[1]), tuple(poses[2]),
                                           speed=a.speed, wait=False)
            except Exception as exc:                      # noqa: BLE001
                # 规划失败（IK 不可达 / 三点近共线 ⇒ 巨半径 / 起点与实测 TCP 不符）是
                # **测量结果**而非安全隐患 ⇒ **响亮地记下来并跳过**，别让一条动作毁掉整档。
                # （判据仍要看得见：段里带 `error`，指标会因此判未实测而不是给分。）
                self._log(f"  ⚠ {label} 笛卡尔规划失败：{exc!r}")
                self.segments.append(Segment(label=label, kind=a.kind, t0=t0,
                                             t1=time.monotonic(), target=None,
                                             extra={"error": repr(exc), "poses": poses}))
                return
            goal = np.asarray(poses[-1][:3], float)
            tt, pp, settled, err = [], [], False, None
            t_start = time.monotonic()
            try:
                while time.monotonic() - t_start < self.arm.move_timeout:
                    got = self.arm.get_tcp().value
                    if got is not None:
                        tt.append(time.monotonic()); pp.append(list(got[:3]))
                    if self._tcp_settled(tt, pp, goal):
                        settled = True
                        break
                    time.sleep(0.01)
            except Exception as exc:                      # noqa: BLE001
                # ⚠ **已知的固件缺陷**：笛卡尔路径偶发**多余的 `0x4E` 应答**
                #   （core 判"应答比请求多、主机与固件已错配"并抛错）。
                #   6 小时浸泡实测 **4/89 = 4.5%**，且**全部**落在笛卡尔段。
                #   ⇒ 那是**被测系统的缺陷**，不该让整轮基准崩掉 —— 记进段里、继续跑。
                #   校验一下：`docs/soak_20260921_0108_full/soak.json` 的 4 条失败记录
                #   全是这一个错误。**要修的是固件/host 的笛卡尔应答配对，不是这里。**
                err = repr(exc)
                self._log(f"  ⚠ {label} 采样出错（记为被测系统缺陷，本轮继续）：{exc!r}")
            t1 = time.monotonic()
            dev_mm = (float(np.linalg.norm(np.asarray(pp[-1], float) - goal) * 1000.0)
                      if pp else None)
        if pp:
            self.trace.tcp_t = np.concatenate([self.trace.tcp_t, np.asarray(tt)])
            self.trace.tcp_p = np.concatenate([self.trace.tcp_p, np.asarray(pp)])
        self.segments.append(Segment(label=label, kind=a.kind, t0=t0, t1=t1, target=None,
                                     extra={"plan_us": getattr(plan, "plan_us", None),
                                            "n_wp": getattr(plan, "n_wp", None),
                                            "settled": settled, "n_tcp": len(pp),
                                            "dev_mm": dev_mm, "poses": poses,
                                            **({"error": err} if err else {})}))
        self._log(f"  {label}: {a.kind} n_wp={getattr(plan,'n_wp',None)} "
                  f"plan_us={getattr(plan,'plan_us',None)} settled={settled} "
                  f"tcp采样={len(pp)} 到位偏差="
                  + ("—" if dev_mm is None else f"{dev_mm:.2f}mm"))
        if a.rest_s:
            self._hold(a.rest_s)

    def _watchdog(self, a: Action) -> None:
        """命令看门狗的跳闸时延。⚠ **常规状态下本测法在原理上观测不到它。**

        固件 `control_loop.c:717` 原文：「`watchdog_kick()` **只在 `mode==MOVE_J` 时
        自动调用**」。而 `movej` 到位后固件**停在 MOVE_J 持位、且每拍自 kick**
        （`watchdog.c` 的 `[B]` 注释：「move_j 受控静止保持持续 kick」）。
        ⇒ 在 MOVE_J 下，"停止发命令"**永远不会**触发看门狗。

        ⛔⛔ **绝不去人为诱发它。** `control_loop.c:1787` 写明触发后的处置是
        「fail-soft 持位（**降刚度、τ=0、允许下垂**）」—— 那是**掉力矩**。
        本机 J4 距上软限只有约 0.01 rad，一次下垂就可能把它推出**锁存线**，
        而那个锁存 `reset` 清不掉（本项目记录过：J4 锁死只能人工推回去）。
        **为测一个时延去冒这个险，不值得。**

        ⇒ 所以这里**如实报未实测并写明理由** —— 而不是报一个读起来像"通过"的
        「未观察到跳闸」。⚠ 「测不到」与「测出来很好」的区分，是本工程的立身之本。

        ⛔ 判据本身必须用 **bit 0x2**（core 解成 `WD_TRIPPED`）；
        按固件拼法搜 `WATCHDOG` 字符串**永远搜不到**（历史事故）。
        """
        WD = 0x2
        t0 = time.monotonic()
        st = self.arm.get_state(refresh=True).value
        mode = str(getattr(st, "mode_name", "")) if st is not None else ""

        if mode == "MOVE_J":
            self.trace.meta["watchdog_latency_s"] = None
            self.trace.meta["watchdog_note"] = (
                f"模式为 `{mode}` ⇒ 固件每拍自 kick（`control_loop.c:717`）"
                f"⇒ 静止等待法**原理上观测不到**跳闸；且触发 = τ=0 下垂，不去诱发")
            self._log(f"  看门狗时延: **未实测**（模式 {mode} 下固件每拍自 kick ⇒ "
                      f"本测法观测不到；触发它要掉力矩，不诱发）")
            self.segments.append(Segment(label=a.label or "watchdog", kind="hold",
                                         t0=t0, t1=time.monotonic(), target=None,
                                         extra={"mode": mode, "latency": None,
                                                "why": "MOVE_J 自 kick，观测不到"}))
            return

        seen = None
        while time.monotonic() - t0 < 5.0:
            st = self.arm.get_state(refresh=True).value
            if st is not None and (int(st.flags) & WD):
                seen = time.monotonic() - t0
                break
            time.sleep(0.02)
        self.trace.meta["watchdog_latency_s"] = seen
        if seen is None:
            self.trace.meta["watchdog_note"] = (
                f"模式 `{mode}`（非 MOVE_J）下等了 5 s 仍未跳闸 ⇒ 可能是别的原因"
                f"在 kick，**不是**「看门狗正常」的证明")
        self._log(f"  看门狗时延: {'未观察到跳闸' if seen is None else f'{seen:.3f}s'}"
                  f"（模式 {mode}）")
        self.segments.append(Segment(label=a.label or "watchdog", kind="hold",
                                     t0=t0, t1=time.monotonic(), target=None,
                                     extra={"mode": mode, "latency": seen}))
        self._hold(0.5)

    def _kin_bench(self, a: Action) -> None:
        """固件自检 `0x49`。⚠ 回执是**连续两帧**（耗时帧 + LINK 帧），core 两帧都收。

        ⛔ 旧实现有两处错，合起来让「固件计算性能」这一维**恒判未实测**：

        1. 读的是 `Msg` **信封**而不是 `.value`。core 把 11 个读型入口包进了信封
           （2.0.0 的破坏性变更），本工程其它地方都写了 `.value`，**只有这里漏了**
           ⇒ `timings` 恒为空 ⇒ 只报出 `{'_frames': 1}`。
        2. 拿 `timings["LOOP1"]` 当「控制拍峰值」。`LOOP1` 是自检里**禁中断单次**测的
           控制步计算量（`kin_runner.c:415-418`），而"控制拍峰值"是固件另一处的
           `loop_k = g_arm.loop_max_cyc/1000`（`litearm.h:307` 注释"历史最长单拍耗时"）。
           真机实测两者差 **10×**：`LOOP1` 20.06 kcycle vs `loop_k` **199** kcycle。
           ⇒ 指标名就叫 `loop_max_kcycle`，取的是**后者**；前者只当旁证记下来。
        """
        try:
            res = self.arm.diag.kin_bench().value          # ⚠ 信封要拆
        except Exception as exc:                           # noqa: BLE001
            self._log(f"  ⚠ 固件自检失败：{exc!r}")
            return
        if res is None:
            self._log("  ⚠ 固件自检没取到值")
            return

        timings = dict(getattr(res, "timings", {}) or {})
        link = dict(getattr(res, "link", {}) or {})
        d: Dict[str, object] = {"_frames": 2 if link else 1,
                                "timing_blocks": sorted(timings)}
        if timings.get("LOOP1"):
            # 旁证：**禁中断单次**测的控制步计算量（≠ 真实整拍耗时）
            d["loop1_isolated_kcycle"] = float(timings["LOOP1"][0]) / 1000.0

        # ⚠⚠ **LINK 帧没到 ⇒ 下面一个数都不许写。**
        #   `link.get("loop_k", 0)` 那类出口会让「读不到」与「真的是 0」长得一模一样
        #   （本项目记录过这次教训：把"计数器恒 0"读成了"没出错"）。
        #   写不写，由「**键到齐没有**」决定，不由"能不能取到一个默认值"决定。
        if link:
            d["loop_max_kcycle"] = float(res.loop_max_kcycle)
            d["overruns"] = int(res.loop_overruns)
            self.trace.meta["crc_errors"] = int(res.crc_errors)
            self.trace.meta["reply_dropped"] = int(res.reply_dropped)
            self.trace.meta["can_tx_fail"] = int(res.can_tx_fail)
            # `[N-12 fix]` 的两个 RX FIFO 溢出计数 —— 溢出 = **静默丢反馈**，
            # 可能整臂 EMERGENCY。键名带数字，解析器从前吃不下。
            self.trace.meta["rx_fifo_lost"] = int(link.get("rxl0", 0)) + int(link.get("rxl1", 0))
        else:
            self._log("  ⚠ 固件自检只回了 1 帧（LINK 段缺失）"
                      "⇒ 控制拍峰值与链路计数一律判**未实测**（不填 0）")

        self.trace.meta["kin_bench"] = {k: v for k, v in d.items() if v is not None}
        self._log(f"  固件自检: {self.trace.meta['kin_bench']}")

    def _gcheck(self, a: Action) -> None:
        """静态 G 对照：在每个静止位形上取 `τ_实测` 与 `model.get_gravity(q)` 比。

        ⚠ **必须用稳态读数**：装载/卸载的瞬态会把偏差读数高估近一倍（本项目踩过，
        并因此得出过"模型只有实测一半"的错误结论）。
        """
        from ..metrics.registry import M_TAU_G_MATCH  # noqa: F401
        poses = a.extra.get("poses") or ["bench", "bench_j2hi", "bench_j2lo"]
        samples = []
        for key in poses:
            tgt = self._pose(key)
            self._gate(tgt, f"gcheck:{key}")
            try:
                self.arm.movej(tgt, speed=0.3)
            except Exception as exc:                      # noqa: BLE001
                self._log(f"  ⚠ gcheck {key} 运动失败：{exc!r}")
                continue
            if not self._at_rest():
                self._log(f"  ⚠ gcheck {key}: 没等到稳态 ⇒ 丢弃该位形"
                          f"（本项目记录过：用瞬态读数会把偏差高估近一倍）")
                continue
            st = self.arm.get_state(refresh=True).value
            q = np.asarray([j.q for j in st.joints], float)
            tau = np.asarray([j.tau for j in st.joints], float)
            try:
                gm = self.arm.model.get_gravity(q).value
            except Exception:                             # noqa: BLE001
                return
            if gm is None:
                return
            gm = np.asarray(gm, float)
            # ⚠ **这里只存原始样本，不做判据** —— 判据要拿 `fc0/fc1`（库仑摩擦）来筛
            #   "模型预测得出来的轴"，而那两个参数要等 `snapshot_identity` 才读回来
            #   （在本动作之后）⇒ 判据放在分析层（`_torque`），那里 identity 齐了。
            #   ⛔ 旧实现就地按 `|τ| > 0.3` 挑轴、取中位数，把两种**完全不同的工况**
            #   混在一起：零位下 J1/J3 的 G 模型本来就是 0，而实测 τ≈0.4 N·m
            #   （**摩擦撑的**）⇒ 比值算出 100%，把整条指标抬到 35%（历史分布中位 6.3%）。
            samples.append({"pose": key,
                            "q": [float(v) for v in q],
                            "tau": [float(v) for v in tau],
                            "g": [float(v) for v in gm]})
            self._log(f"  gcheck {key}: 采到 {len(q)} 轴读数（判据在分析层按 fc0/fc1 筛）")
        if samples:
            self.trace.meta["g_samples"] = samples

    #: 首尾同点位对照用的固定位形 —— 取 `bench`（舒展、非奇异、J4 居中）。
    #: ⚠ 两次必须用**同一个**位形，否则量到的是位形差不是漂移。
    DRIFT_POSE = "bench"

    def _drift_ref(self, a: Action) -> None:
        """记一次**同点位**的落点快照，供 `endur.static_drift_mrad` 前后对比。

        ⚠ 这条指标此前**没有任何数据来源**：`trace.meta['static_drift_mrad']` 全仓
        **没有写入点**，而它的 skip 理由却写着「需 manual/完整档」—— 实际跑哪档都测不到。
        现在由本动作在**每个档的首尾各做一次**（分析层取 `refs[0]` 与 `refs[-1]`）。

        ⚠ 快照取**多帧均值**：单帧受编码器量化（LSB 0.096 mrad）限制，
        而漂移要量的正是 mrad 量级的变化。
        """
        label = a.label or "drift_ref"
        tgt = self._pose(self.DRIFT_POSE)
        self._gate(tgt, f"{label}@{self.DRIFT_POSE}")
        try:
            self.arm.movej(tgt, speed=a.speed)
        except Exception as exc:                          # noqa: BLE001
            self._log(f"  ⚠ 漂移基准 {label}: 运动失败 {exc!r} ⇒ 本次不记")
            return
        if not self._at_rest():
            self._log(f"  ⚠ 漂移基准 {label}: 没等到稳态 ⇒ 本次不记"
                      f"（用瞬态读数会把漂移算错，本项目明确记录过）")
            return
        qs = []
        for _ in range(10):                               # 10 帧 @~50Hz ⇒ 压掉量化噪声
            try:
                qs.append(self._q())
            except Exception:                             # noqa: BLE001 - 读不到就少一帧，不中断
                pass
            time.sleep(0.02)
        if len(qs) < 4:
            self._log(f"  ⚠ 漂移基准 {label}: 只采到 {len(qs)} 帧 ⇒ 本次不记")
            return
        qm = np.asarray(qs, float).mean(axis=0)
        refs = self.trace.meta.setdefault("drift_refs", [])
        refs.append({"label": label, "t": time.monotonic(),
                     "q": [float(v) for v in qm],
                     "target": [float(v) for v in tgt], "n": len(qs)})
        self._log(f"  漂移基准 {label}: 已记（{self.DRIFT_POSE}，{len(qs)} 帧均值）")

    def _manual(self, a: Action) -> None:
        if not self.interactive:
            self._log(f"  ⏭ 需人工动作被跳过（非交互）：{a.manual_hint or a.label}")
            return
        print(f"\n{'='*68}\n【需人工】{a.manual_hint or a.label}\n{'='*68}")
        print("（完成并确认设备已恢复后按回车继续；输入 s 跳过本步）", flush=True)
        try:
            ans = input("> ").strip().lower()
        except EOFError:
            ans = "s"
        if ans == "s":
            self._log(f"  ⏭ 人工步骤被跳过：{a.label}")
            return
        self._mark_rest(a.label or "manual", None)
