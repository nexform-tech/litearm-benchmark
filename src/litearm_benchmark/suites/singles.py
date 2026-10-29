"""单项测试目录 —— 界面上能挑的"各种测试项"。

## 为什么单项也要有目录，而不是让界面现场拼动作

界面里能选到的东西必须**可复现、可写进报告**。动作表是**数据**，放进版本控制；
界面只挑名字。这样"那次 GUI 上跑的结果"与"命令行上重跑一次"是同一件事。

## 它们与档位走**同一条**装配路径

每一项都是普通的一组 :class:`~.base.Action`，经
:func:`litearm_benchmark.session.run_actions` 执行 —— 安全闸（`preflight` 与
`Executor._gate`）、采集、分析、打分**一视同仁**，没有第二条路。

⇒ ⚠ **别在这里写"绕过闸门"的快捷项**。要动臂就得过闸，单项也不例外。

## 运动包线

全部取自 :data:`~.base.POSES` —— 与档位同一套：不碰 J4 的软限上界、不进 URDF 零位奇异、
单轴行程 ≤ 0.8 rad。大范围/贴限位的能力**本工程不做**。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .base import Action


@dataclass(frozen=True)
class SingleItem:
    """一个可单独运行的测试项。"""

    key: str                        #: 稳定标识（写进输出目录名与报告）
    title: str                      #: 界面显示名
    desc: str                       #: 一句话：它测什么、看哪个指标
    actions: Tuple[Action, ...]
    #: 需要的能力，供界面提示（`cart`=笛卡尔 / `sync`=同步 / `log`=300Hz 日志窗）
    needs: Tuple[str, ...] = ()
    #: 该档典型的挂钟时长（秒，实测数量级）—— 只用于界面上的预期提示
    seconds: float = 20.0


SINGLES: Dict[str, SingleItem] = {}


def _add(key: str, title: str, desc: str, actions: List[Action],
         *, needs: Tuple[str, ...] = (), seconds: float = 20.0) -> None:
    SINGLES[key] = SingleItem(key, title, desc, tuple(actions), needs, seconds)


# --------------------------------------------------------------------------- #
# 静态 / 模型
# --------------------------------------------------------------------------- #
_add("gcheck", "静态 G 对照",
     "在 3 个位形上比 `τ_实测` 与 `G(q)` 模型 ⇒ `torque.g_match_pct`（模型可信度）。"
     "⚠ 只在「重力项主导」的轴上计分（零位时 G 本就是 0，实测 τ 是摩擦撑的）。",
     [Action("rest", "起手静止", rest_s=0.3),
      Action("gcheck", "静态 G 对照", n_ticks=200)],
     seconds=25)

# --------------------------------------------------------------------------- #
# 关节运动
# --------------------------------------------------------------------------- #
_add("movej_slow", "往返 movej @0.3",
     "慢速往返（0.5 rad，每轮先回 bench）⇒ 静差、重复性、跟踪的最干净工况。",
     [Action("rest", "起手静止", rest_s=0.3),
      Action("movej", "慢速往返", target="bench_j2lo", via="bench", speed=0.3,
             n_ticks=700, reps=2, rest_s=1.0)],
     needs=("log",), seconds=30)

_add("movej_fast", "高速 movej @1.0",
     "高速大行程（0.8 rad）⇒ 跟踪与越峰的**最坏工况**（工况也是参数，本项目被单工况骗过两次）。",
     [Action("rest", "起手静止", rest_s=0.3),
      Action("movej", "高速往返", target="bench_j2hi", via="bench_j2lo", speed=1.0,
             n_ticks=700, reps=2, rest_s=1.0)],
     needs=("log",), seconds=30)

_add("movej_multi", "多轴联动 @0.6",
     "J1/J3/J5/J7 同时动（1.4 rad）⇒ 重力耦合下的跟踪与力矩 —— 单轴永远看不到这一面。",
     [Action("rest", "起手静止", rest_s=0.3),
      Action("movej", "多轴 Q0→Q1", target="bench_multi", via="bench_multi2",
             speed=0.6, n_ticks=1100, reps=2, rest_s=1.5)],
     needs=("log",), seconds=35)

_add("movej_sync", "同步多轴 (movej_sync)",
     "一条标量 s 驱动全轴 ⇒ **参考级**同拍到位。⚠ 物理同步**测不了**（唯一实测流是 100Hz，"
     "10ms 分辨率低于分散量级）⇒ `sync.stop_spread_ms` 恒判未实测，见 registry。",
     [Action("rest", "起手静止", rest_s=0.3),
      Action("movej_sync", "同步多轴", target="bench_multi", via="bench_multi2",
             speed=0.6, n_ticks=1100, reps=2, rest_s=1.5)],
     needs=("sync", "log"), seconds=35)

# --------------------------------------------------------------------------- #
# 笛卡尔
# --------------------------------------------------------------------------- #
_add("cart_line", "笛卡尔直线 50mm",
     "从当前 TCP 沿 +Y 直走 50mm ⇒ `cart.line_dev_mm`（逐段比各自的连线）与 `settled`。",
     [Action("rest", "起手静止", rest_s=0.3),
      Action("cart_line", "直线 50mm", speed=0.3, n_ticks=1100, rest_s=1.0,
             extra={"dy": 0.05})],
     needs=("cart",), seconds=25)

_add("cart_arc", "笛卡尔圆弧 r40 90°",
     "水平面内 40mm 半径的 90° 弧 ⇒ `cart.arc_dev_mm`（三点定圆的径向偏差）。"
     "⚠ 本项是浸泡里 4.5% 失败率的**复现入口**（多余 `0x4E` 应答）。",
     [Action("rest", "起手静止", rest_s=0.3),
      Action("cart_arc", "圆弧 r40 90°", speed=0.3, n_ticks=1100, rest_s=1.0)],
     needs=("cart",), seconds=25)

# --------------------------------------------------------------------------- #
# 诊断
# --------------------------------------------------------------------------- #
_add("watchdog", "看门狗时延",
     "命令看门狗断连试验。⚠ **常规档位在本测法下观测不到**（MOVE_J 持位时固件每拍自 kick，"
     "`control_loop.c:717`），报告会如实写未实测 + 理由 —— 而**不去诱发它**（触发 = τ=0 下垂）。",
     [Action("rest", "起手静止", rest_s=0.3),
      Action("watchdog", "看门狗时延")],
     seconds=12)

_add("kin_bench", "固件自检 0x49",
     "固件单拍开销自测 ⇒ `compute.loop_max_kcycle`（真实整拍峰值 `loop_k`，"
     "**不是** `LOOP1`，两者差 10×）与链路计数。⚠ 必须**收满两帧**，只收一帧就判未实测。",
     [Action("kin_bench", "固件自检")],
     seconds=15)

_add("drift", "静态漂移对照",
     "同一点位**前后各量一次**（中间跑一段多轴）⇒ `endur.static_drift_mrad`。"
     "间隔越长越说明问题 —— 单项只有约 40 s，只能当**下限**看。",
     [Action("drift_ref", "漂移基准(前)", speed=0.3),
      Action("movej", "中间运动", target="bench_multi", via="bench_multi2",
             speed=0.6, n_ticks=900, reps=2, rest_s=1.0),
      Action("drift_ref", "漂移对照(后)", speed=0.3)],
     needs=("log",), seconds=45)


def describe() -> str:
    """给 CLI / 界面用的清单。"""
    out = []
    for it in SINGLES.values():
        need = f"　需：{'/'.join(it.needs)}" if it.needs else ""
        out.append(f"- `{it.key}` **{it.title}**（~{it.seconds:.0f}s{need}）\n  {it.desc}")
    return "\n".join(out)


def actions_of(key: str) -> List[Action]:
    if key not in SINGLES:
        raise KeyError(f"未知单项 {key!r}；可选 {sorted(SINGLES)}")
    return list(SINGLES[key].actions)
