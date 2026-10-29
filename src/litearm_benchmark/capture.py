"""双流采集 —— 100Hz 实测状态帧 + 300Hz 固件日志。

两条流的分工（来自 :mod:`.instruments` 的仪表边界）：

===========  ==================  ==================================================
流           速率                 内容
===========  ==================  ==================================================
状态帧       100Hz（**不可配置**）  **实测** q / dq / tau / t_mos / t_coil / err
                                + flags / seq / mode / joint_fault
300Hz 日志   300Hz（**上限 2400 拍**） 指令 `q_ref` / 指令 `dq` / 反馈 `tau`
===========  ==================  ==================================================

三个**必须遵守**的采集前提，每一个都踩过：

1. **必须已使能**。未使能时 `log.start` **受理且不报错**，但一拍照录 —— 失败方向是
   **静默的**。所以 :meth:`Trace.require_records` 会把"录到 0 拍"当成硬错误。
2. **一次最多 2400 拍 ≈ 8.0s**，记满自停。长动作要**分窗**采（见 :meth:`DualCapture.window`
   的用法与 :class:`Trace` 的 ``chunks`` 字段）。
3. **日志不是连续流**：它是"记满→读回"的批处理，读回本身有挂钟开销
   （满量程约 863 次往返）。⇒ 别把日志的时间戳当墙钟，**跨流一律靠 `align` 对齐**。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import instruments as I

#: 状态帧挂钩保留的原函数，收尾时必须**逐字还原**（它是 core 的唯一状态帧漏斗）。
_OriginalHook = Callable[[bytes], None]


@dataclass
class Trace:
    """一次运行采集到的全部原始数据（一次运行 = 多个 300Hz 窗口 + 一条连续 100Hz 记录）。"""

    # ---- 100Hz 实测侧（整段连续） ----
    t_status: np.ndarray = field(default_factory=lambda: np.zeros(0))       # (M,) monotonic
    q: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))         # (M,N)
    dq: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    tau: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    t_mos: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    t_coil: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    err: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    flags: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    seq: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    mode: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    # ---- 300Hz 日志侧（按窗口分段） ----
    chunks: List["LogChunk"] = field(default_factory=list)

    # ---- 笛卡尔：TCP 采样（**必须与状态帧同线程**，不许另开线程抢字节流） ----
    # 用法：`move_l(..., wait=False)` → 循环里 `get_tcp()` + `poll_cart()` 直到落定。
    tcp_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tcp_p: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    # ---- 机器身份与窗口标记 ----
    meta: Dict[str, object] = field(default_factory=dict)
    marks: Dict[str, float] = field(default_factory=dict)   # 名字 → t_status 上的时刻

    # ------------------------------------------------------------------ #
    @property
    def n_joints(self) -> int:
        return int(self.q.shape[1]) if self.q.ndim == 2 and self.q.size else 0

    @property
    def duration(self) -> float:
        return float(self.t_status[-1] - self.t_status[0]) if len(self.t_status) > 1 else 0.0

    def mark(self, name: str, t: Optional[float] = None) -> None:
        """在 100Hz 时间轴上打一个标记（动作起点/终点/事件）。

        ⚠ 标记的是 **100Hz 流的时间**；300Hz 日志的时间原点与它**不同**
        （日志从 `log.start` 起算）⇒ 跨流比较前必须 `align`，不要拿标记直接对日志下标。
        """
        self.marks[name] = float(time.monotonic() if t is None else t)

    def require_records(self, n_ticks: int, where: str = "") -> None:
        """断言日志真的录到了 —— 未使能时"受理但一拍照录"是**静默**的。"""
        got = sum(len(c.tick) for c in self.chunks)
        if got < n_ticks:
            raise RuntimeError(
                f"300Hz 日志只录到 {got}/{n_ticks} 拍{'（' + where + '）' if where else ''}。"
                "最常见原因：**未使能**（固件受理 log.start 但不采）或窗口超出 2400 拍上限。")

    def log_arrays(self, chunk_index: int = 0):
        """取第 ``chunk_index`` 个窗口的 ``(t, q_ref, dq_ref, tau)``；无数据返回空数组。"""
        if chunk_index >= len(self.chunks):
            n = self.n_joints
            z = np.zeros((0, n))
            return np.zeros(0), z, z, z
        c = self.chunks[chunk_index]
        t = c.tick.astype(float) / I.CTRL_HZ.value
        return t - t[0], c.q_ref, c.dq_ref, c.tau


@dataclass
class LogChunk:
    """一段 ≤2400 拍的 300Hz 日志。`t` 一律**归零到本段起点**。"""

    tick: np.ndarray
    q_ref: np.ndarray
    dq_ref: np.ndarray
    tau: np.ndarray
    label: str = ""

    @property
    def n_ticks(self) -> int:
        return int(len(self.tick))

    @property
    def duration(self) -> float:
        return self.n_ticks / I.CTRL_HZ.value


class DualCapture:
    """采集器。**进入时挂 100Hz 探头，退出时逐字还原**。

    为什么挂 `_on_status`：它是 core 里**唯一**的状态帧漏斗（`movej` 的到位等待、
    `get_state` 的刷新、`get_tcp` 都从它过）。挂在这里 ⇒ 不额外发帧、不与命令抢链路、
    也不会开出第二个读线程去分吃字节流。
    """

    def __init__(self, arm):
        if not getattr(arm.get_state(refresh=False).value, "enabled", True):
            raise RuntimeError("未使能 —— 300Hz 日志在未使能时**静默不采**，拒绝开始采集。")
        self.arm = arm
        self.trace = Trace()
        self._orig: Optional[_OriginalHook] = None
        self._rec_t: List[float] = []
        self._rec: List[Sequence] = []

    # ------------------------------------------------------------------ #
    def __enter__(self) -> DualCapture:
        a = self.arm._a                                   # noqa: SLF001 - core 的私有 Ack
        self._orig = a._on_status                         # noqa: SLF001
        orig = self._orig

        def hook(payload: bytes) -> None:
            orig(payload)
            st = a.state
            if st is None:
                return
            self._rec_t.append(time.monotonic())
            self._rec.append(st)

        a._on_status = hook                               # noqa: SLF001
        return self

    def __exit__(self, *exc) -> None:
        if self._orig is not None:
            self.arm._a._on_status = self._orig           # noqa: SLF001 - 逐字还原
        self._flush()

    # ------------------------------------------------------------------ #
    def _flush(self) -> None:
        """把挂钩期收下的状态帧转成数组。多次调用是幂等的（只搬新收的）。

        ⚠⚠ **时间轴必须是真实的 `time.monotonic()`，绝不许"接在已有段之后"重排。**
        2026-09-20 真机首次跑 quick 时这里正是这么错的：旧实现把每次 flush 的
        `new_t` 平移成"紧接上一段"，于是**两次 flush 之间的真实间隔被抹掉**，
        `t_status` 成了一条被人为压缩的假时间轴（实测跨度 40.4 s，而运行历时 51 s）。

        后果**不是"精度差一点"而是整体相位错位**：`Segment.t0/t1` 来自
        `time.monotonic()`，而分析按 `t >= s.t1 - 0.6` 在 `t_status` 上选窗口 ——
        两个不同源的时间直接比大小 ⇒ 每个静止段选到的都是**别的动作**的片段，
        静差被算成 **798.769 mrad**（≈ 相邻两个目标之差 0.8 rad），
        而"重复定位精度"也变成跨目标的极差。

        ⚠ 为什么当时没发现：跨流自证量把 `track.*` **整组**判成不可信并拒绝出分 ——
        那条规矩挡住了跨流，但 `static.*` 是**单流**指标，没有自证量挡它。
        """
        if not self._rec:
            return
        new_t = np.asarray(self._rec_t, float)
        t = self.trace.t_status
        if len(t) and len(new_t) and new_t[0] < t[-1]:
            raise RuntimeError(                       # 真实时钟不可能倒退 ⇒ 采集层坏了
                f"状态帧时间轴倒退：{new_t[0]:.6f} < 上一段末尾 {t[-1]:.6f} "
                f"⇒ 时间源不是单调时钟，分析结果不可信")
        self.trace.t_status = np.concatenate([t, new_t]) if len(t) else new_t
        st = self._rec
        cat = lambda f: np.asarray([f(s) for s in st])     # noqa: E731
        def catj(f):
            return np.asarray([[f(j) for j in s.joints] for s in st])
        self.trace.q = _cat(self.trace.q, catj(lambda j: j.q))
        self.trace.dq = _cat(self.trace.dq, catj(lambda j: j.dq))
        self.trace.tau = _cat(self.trace.tau, catj(lambda j: j.tau))
        self.trace.t_mos = _cat(self.trace.t_mos, catj(lambda j: j.t_mos))
        self.trace.t_coil = _cat(self.trace.t_coil, catj(lambda j: j.t_coil))
        self.trace.err = _cat(self.trace.err, np.asarray(
            [[int(j.err) for j in s.joints] for s in st], dtype=int))
        self.trace.flags = _cat(self.trace.flags, cat(lambda s: int(s.flags)))
        self.trace.seq = _cat(self.trace.seq, cat(lambda s: int(s.seq)))
        self.trace.mode = _cat(self.trace.mode, cat(lambda s: int(s.mode)))
        self._rec_t, self._rec = [], []

    # ------------------------------------------------------------------ #
    def window(self, n_ticks: int, label: str = ""):
        """上下文管理器：一段 300Hz 采集窗。

        ::

            with cap.window(900, "step_J2") as w:
                cap.trace.mark("step_start")
                arm.movej(target, speed=0.5)
                cap.trace.mark("step_end")
            # 退出时：等录满 → 读回 → 解析 → 追加到 trace.chunks

        ⚠ ``n_ticks`` 不得超过 :data:`instruments.LOG_CAP_TICKS`；超了直接拒绝
        （固件会回 `ERR{0x2D,0x02}`，与其等到超时不如现在就喊）。
        ⚠ 必须在**本窗之内**发出被观测的动作，否则录到的是一段静止。
        """
        return _Window(self, n_ticks, label)

    def _run_window(self, n_ticks: int, label: str) -> None:
        log = self.arm.log
        log.start(n_ticks)
        self._pending_read = (n_ticks, label, time.monotonic())

    def _close_window(self) -> None:
        n_ticks, label, _ = self._pending_read
        log = self.arm.log
        rd = log.reader()
        rd.wait_for(n_ticks, timeout=max(5.0, n_ticks / I.CTRL_HZ.value * 3 + 3))
        samples = rd.samples()
        if not samples:
            raise RuntimeError(f"窗口 {label!r} 录到 0 拍 —— 检查是否已使能（静默失败）")
        self.trace.chunks.append(LogChunk(
            tick=np.asarray([s.tick for s in samples], dtype=np.int64),
            q_ref=np.asarray([s.q_ref for s in samples], float),
            dq_ref=np.asarray([s.dq for s in samples], float),
            tau=np.asarray([s.tau for s in samples], float),
            label=label,
        ))
        self._flush()


class _Window:
    def __init__(self, cap: DualCapture, n: int, label: str):
        if n <= 0 or n > I.LOG_CAP_TICKS.value:
            raise ValueError(f"n_ticks={n} 越界：1..{I.LOG_CAP_TICKS.value}"
                             f"（超上限固件回 ERR{{0x2D,0x02}}）")
        self.cap, self.n, self.label = cap, n, label

    def __enter__(self):
        self.cap._run_window(self.n, self.label)          # noqa: SLF001
        return self

    def __exit__(self, *exc):
        self.cap._close_window()                          # noqa: SLF001
        return False


# --------------------------------------------------------------------------- #
def _cat(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    if old.size == 0:
        return new
    if new.size == 0:
        return old
    return np.concatenate([old, new], axis=0)


def snapshot_identity(arm) -> Dict[str, object]:
    """把**机器身份**记进运行元数据 —— 没有它，分数不可解释、也不可比。

    记录：固件串、轴数、`ff_mask`、逐轴 `mit_kp/mit_kd/tau_max/q_min/q_max`、
    `fc0/fc1`、`hold_kp_gain`，以及模型状态（override / dirty）。

    ⚠ `fc0/fc1/mit_kp/hold_kp_gain` **必须取设备当天值**：本项目实测过设备值与出厂
    `defaults.c` 差到 1.5×，而静差地板正是它们的函数。
    """
    out: Dict[str, object] = {}
    out["firmware"] = getattr(arm, "firmware", None)
    out["n_joints"] = getattr(arm, "n", None)
    try:
        out["ff_mask"] = int(arm.get_ff_mask())
    except Exception:                                     # noqa: BLE001
        out["ff_mask"] = None

    jp = []
    try:
        for p in arm.params.all_joint_params():
            jp.append(dict(kp=p.kp, kd=p.kd, tau_max=p.tau_max, q_min=p.q_min, q_max=p.q_max))
    except Exception as exc:                              # noqa: BLE001
        out["joint_params_error"] = repr(exc)
    out["joint_params"] = jp

    # 摩擦库仑项（0x2B item10/11）与持位增益（0x2C item18）
    # ⛔ item10/11 走的是 0x2B；`0x2C` 的 item10/11 是 drag_gain/drag_db **且不报错**。
    for key, item in (("fc0", 10), ("fc1", 11)):
        try:
            out[key] = list(arm.get_ff_vec(item).value)
        except Exception:                                 # noqa: BLE001
            out[key] = None
    try:
        out["hold_kp_gain"] = float(arm.get_ff_scalar(18).value)
    except Exception:                                     # noqa: BLE001
        out["hold_kp_gain"] = None
    try:
        ms = arm.model.status().value
        out["model"] = dict(override=ms.override, staged_mask=ms.staged_mask, dirty=ms.dirty)
    except Exception:                                     # noqa: BLE001
        out["model"] = None
    return out
