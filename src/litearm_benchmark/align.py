"""跨流对齐 —— 把 300Hz 的**参考**与 100Hz 的**实测**放到同一条时间轴上。

为什么必须对齐：两条流的时基不同（300Hz 控制拍 vs 100Hz 状态帧），而且 300Hz 日志是
**批处理**的（记满再读回），它的时间戳与墙钟差着一个可变的传输时延。直接相减会得到
「实测领先 9 mrad」这类**对齐伪影**——看着像结论，其实是 bug。

本模块的做法与两个**必须同时报出**的量（都是踩过坑之后加的）：

1. 互相关搜最优时延 δ（在 ±``window`` 内按 ``step`` 扫），取残差 rms 最小者；
2. **自证量** ``control_rms``：故意把 δ 再错开 :data:`CONTROL_OFFSET`（默认 80ms）的残差。
   真实结构下它会明显变差（实测 **3~6×**）；若两流其实无关（噪声），比值 ≈ **1.00×**。
   ⇒ **``ratio`` 接近 1 就意味着"这次对齐没找到真结构"，结果不可用**。

⚠ 三条已知失效模式（都在本项目的历史里真实发生过）：

* **时间原点不统一**：若一条流从"发命令时刻"起算、另一条从"开始记录"起算，
  真偏移会落到搜索窗**之外**，于是 δ 顶在窗边缘、且整表残差是错的。
  ⇒ 本模块强制**两条流各自减掉自己的 ``t[0]``**（:func:`align` 的第一步）。
  用 :attr:`AlignResult.at_edge` 把"δ 顶在窗边"这个症状显式报出来。
* **搜索窗太窄**：真时延曾达到 −30ms 量级，窗要够宽（默认 ±150ms）。
* **把伪影当结论**：所以 ``ok`` 要求 ``ratio`` 达标，报告里**同时**印 δ / rms / control_rms / ratio。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

#: 故意错开的量，用来做自证。取 80ms ≈ 8 个状态帧周期 —— 足够远到结构必然失配，
#: 又近到"慢变信号"不会因此显得一样好。
CONTROL_OFFSET = 0.080

#: 自证量下限。实测真结构 3~6×；离线随机负控 1.00×。取 2.0 作为"这次对齐可信"的门槛。
MIN_RATIO = 2.0

DEFAULT_WINDOW = 0.150
DEFAULT_STEP = 0.002


@dataclass
class AlignResult:
    delta: float            # 最优时延 δ (s)：实测时间轴需要平移 δ 才能对上参考
    rms: float              # δ 下的残差 rms (rad)
    peak: float             # δ 下的残差峰值 (rad)
    control_rms: float      # 错开 CONTROL_OFFSET 后的残差 rms (rad)
    searched: int           # 实际搜过的候选数
    window: float

    @property
    def ratio(self) -> float:
        """自证量 = control_rms / rms。**≲1 说明没找到真结构，本次对齐作废。**"""
        if not np.isfinite(self.rms) or self.rms <= 0:
            return float("nan")
        return self.control_rms / self.rms

    @property
    def at_edge(self) -> bool:
        """δ 顶在搜索窗边缘 ⇒ 真偏移很可能在窗外，结果不可信。"""
        return abs(abs(self.delta) - self.window) < 1.5 * DEFAULT_STEP

    @property
    def ok(self) -> bool:
        return (np.isfinite(self.ratio) and self.ratio >= MIN_RATIO and not self.at_edge
                and np.isfinite(self.rms))

    def why_bad(self) -> str:
        if self.at_edge:
            return (f"δ={self.delta*1e3:+.1f}ms 顶在搜索窗 ±{self.window*1e3:.0f}ms 边缘 "
                    "⇒ 真偏移可能在窗外（检查两条流的时间原点是否都已归零）")
        if not np.isfinite(self.ratio):
            return "自证量无法计算"
        if self.ratio < MIN_RATIO:
            return (f"自证量 {self.ratio:.2f}× < {MIN_RATIO}× ⇒ 没找到真结构，"
                    "该段两流可能无关或参考根本没动")
        return ""

    def render(self) -> str:
        flag = "OK" if self.ok else "不可信"
        return (f"δ={self.delta*1e3:+6.1f}ms  rms={self.rms*1e3:6.3f}mrad  "
                f"峰={self.peak*1e3:7.3f}mrad  自证={self.ratio:4.2f}×  [{flag}]")


def align(t_meas: np.ndarray, q_meas: np.ndarray,
          t_ref: np.ndarray, q_ref: np.ndarray,
          axes: Optional[Sequence[int]] = None,
          *, window: float = DEFAULT_WINDOW, step: float = DEFAULT_STEP,
          min_overlap: int = 30) -> AlignResult:
    """互相关对齐两条流，返回 :class:`AlignResult`。

    :param t_meas: 实测时间戳 (M,)，**会先减掉自己的 t[0]**
    :param q_meas: 实测位置 (M,N)
    :param t_ref:  参考时间戳 (K,)，**会先减掉自己的 t[0]**
    :param q_ref:  参考位置 (K,N)
    :param axes:   参与对齐的轴；``None`` = 全部。**只拿真实运动过的轴参与**
                   （静止轴不携带对齐信息，加进来只会稀释信噪比）。
    """
    t_meas = np.asarray(t_meas, float)
    t_ref = np.asarray(t_ref, float)
    if len(t_meas) < min_overlap or len(t_ref) < min_overlap:
        return AlignResult(float("nan"), float("nan"), float("nan"), float("nan"),
                           0, window)

    # ⚠ 一律归零到各自起点：混用"发命令时刻"与"开始记录时刻"会让真偏移落到窗外。
    tm = t_meas - t_meas[0]
    tr = t_ref - t_ref[0]
    ax = list(range(q_meas.shape[1])) if axes is None else list(axes)
    if not ax:
        return AlignResult(float("nan"), float("nan"), float("nan"), float("nan"),
                           0, window)

    def rms_at(d: float):
        tt = tm + d
        m = (tt >= tr[0]) & (tt <= tr[-1])
        if int(m.sum()) < min_overlap:
            return None
        ref = np.stack([np.interp(tt[m], tr, q_ref[:, a]) for a in ax], axis=1)
        e = q_meas[m][:, ax] - ref
        return float(np.sqrt((e ** 2).mean())), float(np.abs(e).max())

    best = None
    n_searched = 0
    for d in np.arange(-window, window + 1e-9, step):
        r = rms_at(float(d))
        n_searched += 1
        if r and (best is None or r[0] < best[1]):
            best = (float(d), r[0], r[1])
    if best is None:
        return AlignResult(float("nan"), float("nan"), float("nan"), float("nan"),
                           n_searched, window)

    ctrl = rms_at(best[0] + CONTROL_OFFSET)
    return AlignResult(delta=best[0], rms=best[1], peak=best[2],
                       control_rms=(ctrl[0] if ctrl else float("nan")),
                       searched=n_searched, window=window)


def joined(t_meas: np.ndarray, q_meas: np.ndarray,
           t_ref: np.ndarray, q_ref: np.ndarray,
           res: AlignResult):
    """按对齐结果把两流**联合**成同一时间轴上的 ``(t, q_m, q_r)``，只保留重叠段。"""
    tm = np.asarray(t_meas, float); tm = tm - tm[0]
    tr = np.asarray(t_ref, float); tr = tr - tr[0]
    tt = tm + res.delta
    m = (tt >= tr[0]) & (tt <= tr[-1])
    ref = np.stack([np.interp(tt[m], tr, q_ref[:, a]) for a in range(q_ref.shape[1])],
                   axis=1)
    return tt[m], q_meas[m], ref


def moving_axes(t: np.ndarray, q: np.ndarray, frac: float = 0.05) -> list:
    """挑出**真的动过**的轴（行程 > frac × 全轴最大行程）—— 对齐时只该用它们。"""
    if len(q) < 3:
        return []
    span = np.ptp(q, axis=0)
    if span.max() <= 0:
        return []
    return [int(i) for i in np.where(span > frac * span.max())[0]]
