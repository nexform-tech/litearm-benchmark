"""Worker —— **唯一**与机械臂说话的线程。

## ⚠⚠ 为什么必须独占

`litearm_core.Arm` 是**单消费者**设计：`_Ack.expect` / `pump` / `_read_status` 都直接从
同一条串口抢帧。两条读路径会让命令的 ACK 被状态轮询吃掉 —— 表现成随机的
"无应答(超时)"。本项目在 server 上实测过 **21% 的失败率**（广播抢帧），
修法就是"把读收进一个持锁的口"。界面若自己再开一条连接轮询状态，就是重犯那个错。

## ⚠⚠ 而基准的 `run_actions` **自己会开一条 `Arm`**

（`BenchSession.__enter__` 建它、`__exit__` 收尾并把臂 park 回零位。）
⇒ 所以本线程在跑测试**之前必须把手上的连接关掉**、跑完再接回来。
**这不是可选优化，是"跑得起来"的前提** —— 不关的话两条连接会互抢状态帧，
症状是随机的"读不到状态帧"，且极难归因。

## 状态机

    空闲 ──▶ 每 0.5s 读一次状态 + 跑一次 preflight  →  sig_device
    有活 ──▶ 串行执行；跑测试时**先让出串口**     →  sig_log / sig_run_*
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from PyQt5.QtCore import QThread

from ..safety import SafetyAbort, preflight

from .bridge import GuiBridge, PromptBox


@dataclass
class DeviceStatus:
    """设备栏要显示的一切。**取不到就不填**，别给默认值当成"正常"。"""

    online: bool = False
    why: str = ""
    port: str = ""
    firmware: str = ""
    n_joints: int = 0
    enabled: bool = False
    mode: str = ""
    joint_fault: int = 0
    flags: int = 0
    t_max: float = float("nan")
    t_coil_max: float = float("nan")
    margin: float = float("nan")
    tightest: int = 0
    q: List[float] = field(default_factory=list)
    faults: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


#: 固件温度保护在 85 °C 触发，处置是**单轴锁存 + 该轴失能下垂**
#: （`safety_check.c:243-249`）⇒ 界面在这一档就把它标红，别等它跳。
TEMP_ALARM_C = 70.0


class Worker(QThread):
    """唯一持有 `Arm` 的线程。空闲刷新状态；有活时串行执行。"""

    IDLE_TICK_S = 0.5
    #: 人工步骤等操作者应答的上限。超时按 **skip** 处理（应答盒的默认值就是 skip）——
    #: "没应答"与"确认过了"在安全侧**不能同义**。
    PROMPT_TIMEOUT_S = 1800.0

    def __init__(self, bridge: GuiBridge, parent=None) -> None:
        super().__init__(parent)
        self._b = bridge
        self._q: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._stop = threading.Event()
        self._arm = None
        self._limits: List[Tuple[float, float]] = []
        self._port: Optional[str] = None
        self._busy = False

    # ------------------------------------------------------------------ #
    # 线程内部
    # ------------------------------------------------------------------ #
    def _acquire(self) -> bool:
        if self._arm is not None:
            return True
        if not self._port:
            return False
        try:
            from litearm_core import Arm
            self._arm = Arm(port=self._port).connect()
            self._limits = [(p.q_min, p.q_max)
                            for p in self._arm.params.all_joint_params()]
            if not self._limits:
                raise RuntimeError("读不到关节软限位 (0x24) —— 安全闸无法工作")
            return True
        except Exception as exc:                          # noqa: BLE001
            self._arm = None
            self._b.sig_device.emit(DeviceStatus(
                online=False, port=self._port or "", why=f"连接失败：{exc}"))
            return False

    def _release(self) -> None:
        """把连接**让出去** —— `run_actions` 会自己开一条 `Arm`。"""
        if self._arm is not None:
            try:
                self._arm.close()
            except Exception:                             # noqa: BLE001
                pass
            self._arm = None

    def _idle(self) -> None:
        if not self._port or not self._acquire():
            return
        arm = self._arm
        try:
            st = arm.get_state(refresh=True).value
        except Exception as exc:                          # noqa: BLE001
            self._b.sig_device.emit(DeviceStatus(
                online=False, port=self._port, why=f"读状态失败：{exc}"))
            return
        if st is None:
            self._b.sig_device.emit(DeviceStatus(
                online=False, port=self._port, why="读不到状态帧"))
            return
        tm = [float(j.t_mos) for j in st.joints]
        tc = [float(j.t_coil) for j in st.joints]
        s = DeviceStatus(online=True, port=self._port,
                         firmware=getattr(self, "_fw", "") or "",
                         n_joints=len(st.joints),
                         enabled=bool(getattr(st, "enabled", False)),
                         mode=str(getattr(st, "mode_name", "")),
                         joint_fault=int(getattr(st, "joint_fault", 0)),
                         flags=int(st.flags),
                         t_max=max(tm), t_coil_max=max(tc),
                         q=[round(float(j.q), 5) for j in st.joints])
        try:
            rep = preflight(arm, limits=self._limits)
            s.margin = rep.min_margin
            s.tightest = rep.tightest
            s.faults = list(rep.faults)
            s.warnings = list(rep.warnings)
            s.online = s.online and rep.ok is not None
        except SafetyAbort as exc:
            s.faults = [str(exc)]
        self._b.sig_device.emit(s)

    def _prompt(self, hint: str) -> str:
        """**在工作线程上被调用并阻塞** —— 把提示交给主线程弹对话框，等它回话。"""
        box = PromptBox(hint, threading.Event())
        self._b.sig_prompt.emit(box)
        if not box.event.wait(timeout=self.PROMPT_TIMEOUT_S):
            self._b.sig_log.emit(f"⚠ 人工步骤等不到应答 ⇒ 按**跳过**处理：{hint}")
            return "skip"
        return box.answer

    # ------------------------------------------------------------------ #
    def run(self) -> None:                                # noqa: D102 - QThread
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=self.IDLE_TICK_S)
            except queue.Empty:
                if not self._busy:
                    self._idle()
                continue
            try:
                job()
            except Exception as exc:                      # noqa: BLE001
                self._b.sig_error.emit(f"后台任务异常：{exc!r}")
                self._busy = False

    def shutdown(self) -> None:
        self._stop.set()
        self._release()
        self.wait(3000)

    # ------------------------------------------------------------------ #
    # 公开操作（都在工作线程上执行）
    # ------------------------------------------------------------------ #
    def op_set_port(self, port: Optional[str]) -> None:
        def job() -> None:
            self._release()
            self._port = port or None
            if self._port:
                self._acquire()
        self._q.put(job)

    def op_read_identity(self) -> None:
        """只读身份探测（固件串/轴数/mit_kp…）—— 给"设备"页用。"""
        def job() -> None:
            if not self._acquire():
                return
            from ..capture import snapshot_identity
            try:
                ident = snapshot_identity(self._arm)
                self._b.sig_task.emit("identity", ident)
            except Exception as exc:                      # noqa: BLE001
                self._b.sig_error.emit(f"身份读取失败：{exc!r}")
        self._q.put(job)

    def op_run(self, *, actions: Sequence, label: str, outdir: str,
               manual_ok: bool = True) -> None:
        """跑一组动作。**先让出串口**，跑完再接回来。"""
        from ..session import run_actions

        def job() -> None:
            self._busy = True
            self._b.sig_run_started.emit(label)
            self._release()                              # ⚠ 关键：让出串口
            res = None
            t0 = time.monotonic()
            try:
                res = run_actions(actions, label=label, port=self._port, outdir=outdir,
                                  interactive=manual_ok, on_log=self._b.sig_log.emit,
                                  on_prompt=self._prompt if manual_ok else None)
                self._b.sig_log.emit(f"✅ {label} 完成（{time.monotonic()-t0:.0f}s）："
                                     f"{res.summary()}")
            except SafetyAbort as exc:
                # ⚠ 安全闸拒绝**不是失败**，是它该有的行为 —— 如实转达，绝不绕过
                self._b.sig_error.emit(f"⛔ 安全闸拒绝动臂：{exc}")
            except Exception as exc:                      # noqa: BLE001
                self._b.sig_error.emit(f"⛔ {label} 失败（{time.monotonic()-t0:.0f}s）：{exc!r}")
            finally:
                self._busy = False
                self._b.sig_run_finished.emit(res)
                self._acquire()                          # 接回来
        self._q.put(job)

    def op_task(self, name: str, fn: Callable[[], object]) -> None:
        """跑一个**纯离线**任务（自检 / 列历史 / 重算 / 对比），把结果发回主线程。"""
        def job() -> None:
            try:
                self._b.sig_task.emit(name, fn())
            except Exception as exc:                      # noqa: BLE001
                self._b.sig_error.emit(f"{name} 失败：{exc!r}")
        self._q.put(job)
