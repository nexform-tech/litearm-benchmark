"""四个档位 —— 动作列表是纯数据，执行器只有一份（:class:`.base.Executor`）。

======================  ==========  ================================================
档位                     目标时长    覆盖
======================  ==========  ================================================
``quick``                ~5 min      静差/重复性/跟踪/抖动/看门狗/固件自检 —— CI 用
``standard``             ~15 min     加笛卡尔、同步、多轴、更多静止位形
``full``                 ~40 min     加长时间稳定性、热、多速度档、更多重复
``manual_load``          视操作      需人工装配重；**不进默认档位**，需显式点名
======================  ==========  ================================================

⚠ 所有动作都在**安全台架包线**内（位形取自 :data:`..suites.base.POSES`）：
单轴行程 ≤ 0.8 rad、不碰 J4 软限上界、不进奇异。大范围/贴限位的能力**本工程不做**。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .base import MANUAL, Action

#: 档位名 → (动作列表, 挂钟预算秒)
TIERS: Dict[str, tuple] = {}


def _quick() -> List[Action]:
    return [
        Action("rest", "起手静止", rest_s=0.5),
        Action("gcheck", "静态 G 对照", n_ticks=200),
        # 小位移重复 3 次：**同一目标**才能算重复定位精度
        Action("movej", "小位移@0.3", target="bench_j2lo", speed=0.3, n_ticks=700,
               reps=3, rest_s=1.0),
        # 高速档：这一档的残余最大，是跟踪与越峰的最坏工况
        Action("movej", "小位移@1.0", target="bench_j2hi", speed=1.0, n_ticks=700,
               reps=2, rest_s=1.0),
        # 中位移：给整定时间/上升时间一个"正常档"
        Action("movej", "中位移@0.5", target="bench", speed=0.5, n_ticks=900,
               reps=2, rest_s=1.5),
        Action("movej", "反向@0.5", target="bench_j2lo", speed=0.5, n_ticks=900,
               rest_s=1.5),
        Action("rest", "收尾静止", rest_s=1.0),
        Action("watchdog", "看门狗时延"),
        Action("kin_bench", "固件自检"),
    ]


def _standard() -> List[Action]:
    a = _quick()
    a += [
        # 多轴联动 —— 跟踪/力矩的真实工况（单轴永远看不到重力耦合）
        Action("movej", "多轴 Q0→Q1", target="bench_multi", speed=0.6, n_ticks=1100,
               reps=2, rest_s=1.5),
        Action("movej", "多轴 Q1→Q2", target="bench_multi2", speed=0.6, n_ticks=1100,
               reps=2, rest_s=1.5),
        # 同步：与前面的逐轴 movej 同目标同速度，直接可比
        Action("movej_sync", "同步多轴", target="bench_multi", speed=0.6,
               n_ticks=1100, reps=2, rest_s=1.5),
        # 笛卡尔
        Action("cart_line", "直线 50mm", speed=0.3, n_ticks=1100, rest_s=1.0,
               extra={"dy": 0.05}),
        Action("cart_arc", "圆弧 r40 90°", speed=0.3, n_ticks=1100, rest_s=1.0),
        # 更多静止位形 ⇒ 静差/G 对照跨构型
        Action("gcheck", "静态 G 对照(3 位形)", n_ticks=200,
               extra={"poses": ["bench", "bench_j2lo", "bench_j2hi", "bench_multi"]}),
        Action("rest", "收尾静止", rest_s=1.0),
    ]
    return a


def _full() -> List[Action]:
    a = _standard()
    a += [
        # 速度档扫描：跟踪/越峰/纹波随速度的曲线（**工况也是参数**，本项目被单工况骗过两次）
        Action("movej", "速度档 0.2", target="bench_j2hi", speed=0.2, n_ticks=1400,
               rest_s=1.0),
        Action("movej", "速度档 0.4", target="bench_j2lo", speed=0.4, n_ticks=1200,
               rest_s=1.0),
        Action("movej", "速度档 0.8", target="bench_j2hi", speed=0.8, n_ticks=1000,
               rest_s=1.0),
        Action("movej", "速度档 1.0", target="bench_j2lo", speed=1.0, n_ticks=1000,
               rest_s=1.0),
        # 长时间稳定性：seq 连续性 + 热斜率（≥2min 才有意义）
        Action("movej", "耐久循环", target="bench_multi", speed=0.5, n_ticks=600,
               reps=14, rest_s=0.3),
        Action("wait", "热平衡", wait_s=20.0),
        Action("rest", "耐久后静止", rest_s=1.0),
        Action("gcheck", "耐久后 G 对照", n_ticks=200),
        Action("kin_bench", "固件自检(耐久后)"),
    ]
    return a


def _manual_load() -> List[Action]:
    """需人工装配重的档位。**不与其它档位同时跑** —— 它改变机器状态。"""
    return [
        Action("manual", "装上配重块（法兰外伸，注意线缆）",
               manual_hint="在法兰上装好配重块，确认紧固、线缆不被拉扯。",
               requires_manual=True, n_ticks=200),
        # 装载后**必须**把 payload 写进固件前馈，否则腕部静差下垂、积分憋着，
        # 读数会明显偏离稳态（本项目因此误判过"模型只有实测一半"）。
        Action("manual", "把配重写进固件前馈（0x28 item4/5）",
               manual_hint="用 set_payload(mass, com) 写入；**拆配重前要设回 0**。",
               requires_manual=True, n_ticks=200),
        Action("rest", "装载后静止", rest_s=1.0),
        Action("gcheck", "装载后 G 对照", n_ticks=200),
        Action("movej", "装载后小位移@0.3", target="bench_j2lo", speed=0.3,
               n_ticks=700, reps=3, rest_s=1.0),
        Action("movej", "装载后中位移@0.5", target="bench", speed=0.5, n_ticks=900,
               reps=2, rest_s=1.5),
        Action("manual", "**拆下**配重并把前馈设回 0",
               manual_hint="先把固件前馈设回 0，再拆配重 —— 顺序反了会有一次过补偿上跳。",
               requires_manual=True, n_ticks=200),
        Action("rest", "卸载后静止", rest_s=1.0),
        Action("gcheck", "卸载后 G 对照", n_ticks=200),
    ]


TIERS["quick"] = (_quick(), 6 * 60)
TIERS["standard"] = (_standard(), 18 * 60)
TIERS["full"] = (_full(), 45 * 60)
TIERS["manual_load"] = (_manual_load(), None)      # 人工档不设预算

DEFAULT_TIER = "standard"


def actions_of(tier: str) -> List[Action]:
    if tier not in TIERS:
        raise KeyError(f"未知档位 {tier!r}；可选 {sorted(TIERS)}")
    return list(TIERS[tier][0])


def budget_of(tier: str):
    return TIERS[tier][1]


def describe() -> str:
    """给 CLI 的 `list` 用：打印每档的动作清单与是否需人工。"""
    out = []
    for name, (acts, budget) in TIERS.items():
        mins = "—" if budget is None else f"~{budget // 60} min"
        man = "**需人工**" if any(a.is_manual for a in acts) else "无人值守"
        out.append(f"\n## {name}（{mins}，{man}，共 {len(acts)} 条动作）\n")
        for a in acts:
            tgt = f" → {a.target}" if a.target else ""
            rep = f" ×{a.reps}" if a.reps > 1 else ""
            hint = f" —— {a.manual_hint}" if a.is_manual else ""
            out.append(f"- `{a.kind}` {a.label}{tgt}{rep}{hint}")
    return "\n".join(out)
