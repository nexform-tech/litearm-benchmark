#!/usr/bin/env python3
"""循环浸泡测试 —— 连续跑若干小时，每轮一个档位，**轮间做健康检查**。

用法::

    python scripts/soak.py --hours 6 --tier full

## 为什么"轮间检查"是必须的，而不是保险

固件在 **85 °C** 触发温度保护，处置是 `safety_check.c:243-249`：

    flags |= ARM_FLAG_TEMP_WARNING;
    g_arm.joint_fault |= (1u << i);        /* [G7] 单关节锁存 */
    ... 会失能过热轴 (保护电机), 该轴将失力下垂。

⇒ 过热是**两件最坏的事同时发生**：`joint_fault` 锁存（`reset` 清不掉）
**加上**该轴掉力矩下垂。而本机 J4 距锁存线只有约 0.06 rad，
一次下垂就可能把它推出去 —— 所以温度闸必须设得**远早于** 85 °C。

同理，任何一轮把臂带偏之后，**后面每一轮都该被安全闸拦住**，而不是接着跑。

## 三条永不让步的

1. **任何 `joint_fault` ⇒ 立刻停并停在零位**（锁存不受 enabled 门控、reset 清不掉）；
2. **温度超 `TEMP_ABORT_C` ⇒ 立刻停**（留 15 °C 余量给固件的 5 拍去抖窗口）；
3. **收尾**：任何退出路径都走 `park()`（回零位 + 保持使能），**绝不 `disable()`**。

⚠ 本脚本**不碰** `save_params`，不做 DFU，不改固件参数 —— 它只跑基准。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from litearm_benchmark import run                      # noqa: E402
from litearm_benchmark.safety import SafetyAbort, park, preflight   # noqa: E402

#: 固件阈值是 `temp_warning_c = 85.0`（`defaults.c:51`）。
#: 停跑线留 **15 °C 余量**：固件要连续 5 拍（≈17ms）才锁存，温度不可能在毫秒内跳 15 °C。
TEMP_ABORT_C = 70.0
TEMP_WARN_C = 60.0

#: 连续这么多轮失败就放弃（避免在一个坏状态上空转）。
MAX_CONSECUTIVE_FAILURES = 3


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_health(port: Optional[str]) -> Dict[str, object]:
    """轮间健康检查。**只读**，不开运动。"""
    from litearm_core import Arm
    out: Dict[str, object] = {"ok": True, "faults": [], "warns": [],
                              "t_mos_max": None, "t_coil_max": None, "flags": None}
    arm = Arm(port=port).connect()
    try:
        st = arm.get_state(refresh=True).value
        if st is None:
            out["ok"] = False
            out["faults"].append("读不到状态帧")
            return out

        q = [j.q for j in st.joints]
        tm = [float(j.t_mos) for j in st.joints]
        tc = [float(j.t_coil) for j in st.joints]
        out["t_mos_max"], out["t_coil_max"] = max(tm), max(tc)
        out["flags"] = int(st.flags)
        out["q"] = [round(v, 4) for v in q]

        if getattr(st, "joint_fault", 0):
            axes = [i + 1 for i in range(len(q)) if st.joint_fault & (1 << i)]
            out["ok"] = False
            out["faults"].append(f"joint_fault=0x{st.joint_fault:04X} 锁存轴 {axes}")

        # ⚠ 按 **flag_names 里的名字**找，别硬编码位号 —— 布局一变硬编码就静默错。
        #   （本项目记录过：按固件拼法搜 "WATCHDOG" 永远搜不到。）
        names = [str(n).upper() for n in (getattr(st, "flag_names", None) or [])]
        for bit, nm in enumerate(names):
            if (int(st.flags) >> bit) & 1:
                if "TEMP" in nm:
                    out["ok"] = False
                    out["faults"].append(f"固件已置温度告警（{nm}）")
                else:
                    out["warns"].append(f"flags 置位：{nm}")

        hottest = max(out["t_mos_max"], out["t_coil_max"])
        if hottest > TEMP_ABORT_C:
            out["ok"] = False
            out["faults"].append(f"温度 {hottest:.1f}°C > 停跑线 {TEMP_ABORT_C}°C"
                                 f"（固件在 85°C 会锁存 + 下垂）")
        elif hottest > TEMP_WARN_C:
            out["warns"].append(f"温度 {hottest:.1f}°C 偏高（停跑线 {TEMP_ABORT_C}°C）")

        # 每轮都重跑安全前置：一旦某轮把臂带偏，后面每轮都该被它拦住
        lim = [(p.q_min, p.q_max) for p in arm.params.all_joint_params()]
        rep = preflight(arm, limits=lim)
        if not rep.ok:
            out["ok"] = False
            out["faults"].extend(rep.faults)
        out["warns"].extend(rep.warnings)
        out["margin"] = rep.min_margin
    except Exception as exc:                            # noqa: BLE001
        out["ok"] = False
        out["faults"].append(f"健康检查自身失败：{exc!r}")
    finally:
        try:
            arm.close()
        except Exception:                               # noqa: BLE001
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="循环浸泡测试")
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--tier", default="full")
    ap.add_argument("--port", default=None)
    ap.add_argument("--max-rounds", type=int, default=400)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(
        "docs", f"soak_{time.strftime('%Y%m%d_%H%M')}_{args.tier}")
    os.makedirs(outdir, exist_ok=True)
    deadline = time.monotonic() + args.hours * 3600.0
    rounds: List[Dict[str, object]] = []
    consec_fail = 0
    stop_reason = ""

    _log(f"浸泡开始：档位 {args.tier}，目标 {args.hours:g} 小时，输出 {outdir}")
    _log(f"温度闸：告警 {TEMP_WARN_C}°C / 停跑 {TEMP_ABORT_C}°C（固件在 85°C 锁存+下垂）")

    for rnd in range(1, args.max_rounds + 1):
        if time.monotonic() >= deadline:
            stop_reason = f"到时间（{args.hours:g} 小时）"
            break

        h = check_health(args.port)
        _log(f"第 {rnd} 轮 健康检查：ok={h['ok']} "
             f"温度 {h.get('t_mos_max')}/{h.get('t_coil_max')}°C "
             f"裕度 {h.get('margin')}")
        for f in h["faults"]:
            _log(f"  ⛔ {f}")
        for w in h["warns"]:
            _log(f"  ⚠ {w}")
        if not h["ok"]:
            stop_reason = f"健康检查不通过（{h['faults']}）"
            break

        t0 = time.monotonic()
        try:
            res = run(args.tier, port=args.port, outdir=None, interactive=False)
            card = res.card
            rec = {"round": rnd, "ok": True, "seconds": round(time.monotonic() - t0, 1),
                   "total": card.total, "grade": card.grade,
                   "unmeasured": len([k for k, v in res.raw.values.items() if v is None]),
                   "t_mos_max": h.get("t_mos_max"), "margin": h.get("margin")}
            consec_fail = 0
            _log(f"  第 {rnd} 轮完成：{card.total if card.total is None else round(card.total, 1)}"
                 f" / {card.grade}　用时 {rec['seconds']}s")
        except Exception as exc:                        # noqa: BLE001
            consec_fail += 1
            rec = {"round": rnd, "ok": False, "seconds": round(time.monotonic() - t0, 1),
                   "error": repr(exc), "trace": traceback.format_exc()[-800:],
                   "t_mos_max": h.get("t_mos_max")}
            _log(f"  ⛔ 第 {rnd} 轮失败（连续 {consec_fail} 次）：{exc!r}")
            if consec_fail >= MAX_CONSECUTIVE_FAILURES:
                stop_reason = f"连续 {consec_fail} 轮失败"
                rounds.append(rec)
                break
        rounds.append(rec)
        with open(os.path.join(outdir, "soak.json"), "w", encoding="utf-8") as fh:
            json.dump({"config": vars(args), "stop_reason": stop_reason,
                       "rounds": rounds}, fh, ensure_ascii=False, indent=1)

    # ---- 收尾：**绝不 disable** ----
    _log(f"停止原因：{stop_reason or '（跑满最大轮数）'}")
    try:
        from litearm_core import Arm
        arm = Arm(port=args.port).connect()
        try:
            park(arm)
            st = arm.get_state(refresh=True).value
            _log(f"收尾完成：enabled={getattr(st, 'enabled', '?')} "
                 f"joint_fault=0x{getattr(st, 'joint_fault', 0):04X}")
        finally:
            arm.close()
    except Exception as exc:                            # noqa: BLE001
        _log(f"⚠ 收尾失败（臂可能仍在保持，未失能）：{exc!r}")

    okr = [r for r in rounds if r.get("ok")]
    _log(f"共 {len(rounds)} 轮，成功 {len(okr)}，输出 {outdir}/soak.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
