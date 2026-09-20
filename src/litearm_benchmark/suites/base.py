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
from ..metrics.analysis import Segment

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
        """每一次动臂前的**限位闸** —— 裕度不足直接抛，不给"就这一次"留口子。"""
        from ..safety import SafetyAbort
        n = len(target)
        lim = self.s.limits
        worst = min(min(target[i] - lim[i][0], lim[i][1] - target[i]) for i in range(n))
        if worst < 0.02:
            raise SafetyAbort(f"动作 {label!r} 的目标位形裕度只有 {worst:+.4f} rad ⇒ 拒绝执行")
        cur = self.trace.q[-1] if self.trace.q.size else np.asarray(target)
        cur_margin = min(min(float(cur[i]) - lim[i][0], lim[i][1] - float(cur[i]))
                         for i in range(min(len(cur), n)))
        if cur_margin < 0.01:
            raise SafetyAbort(f"动作 {label!r} 的**起点**裕度只有 {cur_margin:+.4f} rad ⇒ 拒绝执行")

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

    def _q(self) -> np.ndarray:
        st = self.arm.get_state(refresh=True).value
        return np.asarray([j.q for j in st.joints], float)

    def _mark_rest(self, label: str, target: Optional[Sequence[float]]) -> None:
        t0 = self.trace.t_status[-1] if self.trace.t_status.size else time.monotonic()
        time.sleep(0.2)
        self.segments.append(Segment(label=label, kind="rest", t0=t0,
                                     t1=time.monotonic(),
                                     target=list(target) if target is not None else None))

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
            time.sleep(a.wait_s or 1.0)
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
        raise ValueError(f"未知动作类型 {kind!r}")

    def _joint(self, a: Action) -> None:
        tgt = self._pose(a.target or "bench")
        for rep in range(a.reps):
            label = f"{a.label or a.kind}#{rep}" if a.reps > 1 else (a.label or a.kind)
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
                self.segments.append(seg)
            self._log(f"  {label}: {'sync' if a.kind=='movej_sync' else 'movej'} "
                      f"@speed {a.speed} → {np.round(tgt, 3).tolist()}")
            if a.rest_s:
                time.sleep(a.rest_s)
                self._mark_rest(label + ":rest", tgt)

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
            poses = [list(c) + rpy, list(via) + rpy, goal]
        label = a.label or a.kind
        with self.cap.window(a.n_ticks, label):
            t0 = time.monotonic()
            if a.kind == "cart_line":
                plan = self.arm.move_l(tuple(poses[0]), speed=a.speed, wait=False)
            else:
                plan = self.arm.move_c(tuple(poses[0]), tuple(poses[1]), tuple(poses[2]),
                                       speed=a.speed, wait=False)
            tt, pp, settled = [], [], False
            t_start = time.monotonic()
            while time.monotonic() - t_start < self.arm.move_timeout:
                got = self.arm.get_tcp().value
                if got is not None:
                    tt.append(time.monotonic()); pp.append(list(got[:3]))
                r = self.arm.poll_cart()
                if r is not None and getattr(r, "settled", False):
                    settled = True
                    break
                time.sleep(0.01)
            t1 = time.monotonic()
        if pp:
            self.trace.tcp_t = np.concatenate([self.trace.tcp_t, np.asarray(tt)])
            self.trace.tcp_p = np.concatenate([self.trace.tcp_p, np.asarray(pp)])
        self.segments.append(Segment(label=label, kind=a.kind, t0=t0, t1=t1, target=None,
                                     extra={"plan_us": getattr(plan, "plan_us", None),
                                            "n_wp": getattr(plan, "n_wp", None),
                                            "settled": settled, "n_tcp": len(pp),
                                            "poses": poses}))
        self._log(f"  {label}: {a.kind} n_wp={getattr(plan,'n_wp',None)} "
                  f"plan_us={getattr(plan,'plan_us',None)} settled={settled} tcp采样={len(pp)}")
        if a.rest_s:
            time.sleep(a.rest_s)

    def _watchdog(self, a: Action) -> None:
        """看门狗时延：**停止发命令**，等 `flags & 0x2`（`WD_TRIPPED`）置位。

        ⛔ 判据必须用 **bit 0x2**。core 把这个 bit 解成 `WD_TRIPPED`，
        按固件里的拼法去搜 `WATCHDOG` 字符串**永远搜不到**（本项目的一起假警报）。
        """
        WD = 0x2
        t0 = time.monotonic()
        seen = None
        while time.monotonic() - t0 < 5.0:
            st = self.arm.get_state(refresh=True).value
            if st is not None and (int(st.flags) & WD):
                seen = time.monotonic() - t0
                break
            time.sleep(0.02)
        self.trace.meta["watchdog_latency_s"] = seen
        self._log(f"  看门狗时延: {'未观察到跳闸' if seen is None else f'{seen:.3f}s'}")
        self.segments.append(Segment(label=a.label or "watchdog", kind="hold",
                                     t0=t0, t1=time.monotonic(), target=None,
                                     extra={"latency": seen}))
        time.sleep(0.5)

    def _kin_bench(self, a: Action) -> None:
        try:
            res = self.arm.diag.kin_bench()
        except Exception as exc:                          # noqa: BLE001
            self._log(f"  ⚠ 固件自检失败：{exc!r}")
            return
        d = {}
        timings = getattr(res, "timings", {}) or {}
        loop = timings.get("LOOP1")
        if loop:
            d["loop_max_kcycle"] = float(max(loop)) / 1000.0
        d["overruns"] = getattr(res, "loop_overruns", None)
        d["_frames"] = 2 if timings else 1
        self.trace.meta["kin_bench"] = {k: v for k, v in d.items() if v is not None}
        for attr in ("crc_errors", "reply_dropped", "loop_overruns"):
            v = getattr(res, attr, None)
            if v is not None:
                self.trace.meta[attr] = v
        self._log(f"  固件自检: {self.trace.meta.get('kin_bench')}")

    def _gcheck(self, a: Action) -> None:
        """静态 G 对照：在每个静止位形上取 `τ_实测` 与 `model.get_gravity(q)` 比。

        ⚠ **必须用稳态读数**：装载/卸载的瞬态会把偏差读数高估近一倍（本项目踩过，
        并因此得出过"模型只有实测一半"的错误结论）。
        """
        from ..metrics.registry import M_TAU_G_MATCH  # noqa: F401
        poses = a.extra.get("poses") or ["bench", "bench_j2hi", "bench_j2lo"]
        errs = []
        for key in poses:
            tgt = self._pose(key)
            self._gate(tgt, f"gcheck:{key}")
            try:
                self.arm.movej(tgt, speed=0.3)
            except Exception as exc:                      # noqa: BLE001
                self._log(f"  ⚠ gcheck {key} 运动失败：{exc!r}")
                continue
            if not self._at_rest():
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
            n = 0
            for i in range(len(q)):
                if abs(tau[i]) > 0.3:                     # 只在有力矩的轴上比，免得除以噪声
                    errs.append(abs(tau[i] - gm[i]) / abs(tau[i]))
                    n += 1
            self._log(f"  gcheck {key}: {n} 轴参与")
        if errs:
            self.trace.meta["g_match_pct"] = float(np.median(errs) * 100.0)

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
