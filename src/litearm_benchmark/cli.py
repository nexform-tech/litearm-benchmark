"""命令行 —— ``litearm-bench``。

⚠ 约定照本家族：**argparse + 位置参数 ``action`` + ``choices``**（家族里 `add_parser`
零命中，不用子命令）；``--kebab-case``、help 全中文、``main() -> int``。

⚠ 安全默认：``run`` **会动真机**，故必须显式 ``--go`` 才执行；不给 ``--go`` 时只打印
将要跑的动作清单然后退出。这与 `examples/` 的 ``--go`` 惯例一致
（core 的 CLI 用 ``--dry-run`` 是"拒绝危险"，方向相反，别混）。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import __version__, instruments as I


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="litearm-bench",
        description="LiteArm 机械臂综合性能基准 —— 分维度打分 + 总分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  litearm-bench list\n"
               "  litearm-bench doctor\n"
               "  litearm-bench run --tier quick --go\n"
               "  litearm-bench score --dir docs/bench_20260920_2100_quick\n"
               "  litearm-bench compare --a docs/runA --b docs/runB\n")
    ap.add_argument("action", nargs="?", default="status",
                    choices=["status", "list", "doctor", "run", "score", "compare"],
                    help="要做什么（默认 status）")
    ap.add_argument("--tier", default="standard",
                    choices=["quick", "standard", "full", "manual_load"],
                    help="档位（默认 standard）")
    ap.add_argument("--port", default=None,
                    help="机械臂 CDC 串口（如 /dev/ttyACM0）。缺省按 VID:PID 1d50:606f 自动查找")
    ap.add_argument("--out", default=None,
                    help="落盘目录（默认 docs/bench_<日期>_<时刻>_<档位>）")
    ap.add_argument("--dir", default=None, help="score：要重算的 run 目录")
    ap.add_argument("--a", dest="dir_a", default=None, help="compare：基准 run 目录")
    ap.add_argument("--b", dest="dir_b", default=None, help="compare：对照 run 目录")
    ap.add_argument("--dry-run", action="store_true",
                    help="走离线桩传输（不连真机）。⚠ 桩不建模运动 ⇒ 运动类指标会**未实测**")
    ap.add_argument("--go", action="store_true",
                    help="确认执行会**动真机**的动作（run 档位必需；不给则只打印计划）")
    ap.add_argument("--no-interactive", action="store_true",
                    help="禁止需要人工介入的动作（含人工动作的档位会被拒绝）")
    ap.add_argument("--quiet", action="store_true", help="只打印总分")
    ap.add_argument("--version", action="version", version=f"litearm-benchmark {__version__}")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    act = args.action

    if act == "status":
        print(f"litearm-benchmark {__version__}")
        print(f"指标 {len(_registry().ALL_METRICS)} 项，维度 {len(_registry().DIMENSION_ORDER)} 个")
        bad = _registry().audit_curves()
        print("指标曲线自检：" + ("✅ 全部合规" if not bad else f"⛔ {len(bad)} 处问题"))
        for b in bad:
            print("   " + b)
        print(f"档位：" + ", ".join(f"{k}" for k in _tiers().TIERS))
        return 0

    if act == "list":
        print("# litearm-benchmark 档位与动作")
        print(_tiers().describe())
        return 0

    if act == "doctor":
        return _cmd_doctor(args)

    if act == "run":
        return _cmd_run(args)

    if act == "score":
        if not args.dir:
            sys.exit("score 需要 --dir <run 目录>")
        from .replay import rescore
        card = rescore(args.dir)
        _print_card(card, quiet=args.quiet)
        return 0

    if act == "compare":
        if not args.dir_a or not args.dir_b:
            sys.exit("compare 需要 --a <目录> --b <目录>")
        from .replay import compare
        md = compare(args.dir_a, args.dir_b)
        out = os.path.join(args.dir_b, "COMPARE.md")
        try:
            open(out, "w", encoding="utf-8").write(md)
        except OSError:
            out = None
        print(md)
        if out:
            print(f"\n（已写入 {out}）")
        return 0

    return 2


# --------------------------------------------------------------------------- #
def _cmd_run(args) -> int:
    tier = args.tier
    manual = any(a.is_manual for a in _tiers().actions_of(tier))
    if not args.go and not args.dry_run:
        print(f"⚠ 档位 **{tier}** 会**动真机**。将要执行的动作：")
        print(_tiers().describe().split(f"## {tier}")[-1].split("\n## ")[0].rstrip())
        print("\n确认无误后加 `--go` 重跑；或加 `--dry-run` 走离线桩。")
        return 1
    if manual and args.no_interactive:
        sys.exit(f"档位 {tier!r} 含需人工介入的动作，与 --no-interactive 冲突")

    from .session import run
    outdir = args.out or os.path.join(
        "docs", f"bench_{_stamp()}_{tier}")

    def log(m: str) -> None:
        if not args.quiet:
            print(m, flush=True)

    print(f"litearm-benchmark {__version__}　档位 {tier}　输出 {outdir}")
    res = run(tier, port=args.port, outdir=outdir, dry_run=args.dry_run,
              interactive=not args.no_interactive, on_log=log)
    _print_card(res.card, quiet=args.quiet)
    print(f"\n落盘：{outdir}（trace.npz / raw.json / metrics.csv / REPORT.md）")
    return 0


def _cmd_doctor(args) -> int:
    """环境与口径自检。**不动机器人**。"""
    print(f"litearm-benchmark {__version__}\n")
    print("## 指标曲线自检")
    bad = _registry().audit_curves()
    if bad:
        print(f"⛔ {len(bad)} 处不合规：")
        for b in bad:
            print("   " + b)
    else:
        print("✅ 全部合规（锚点升序、分数在 0~100、单调方向与 direction 一致）")
    ph = _registry().unresolved_placeholders()
    print(f"\n## 仍靠占位阈值的指标（{len(ph)} 项，分数只有相对意义）")
    for k in ph:
        print(f"   - {k}")

    print("\n## 仪表边界")
    print(I.describe())

    print("\n## 环境")
    try:
        import litearm_core
        print(f"   litearm-core {litearm_core.__version__}  ({litearm_core.__file__})")
    except Exception as exc:                              # noqa: BLE001
        print(f"   ⛔ 无法 import litearm_core：{exc!r}")
    try:
        from litearm_core import find_cdc_port
        p = find_cdc_port()
        print(f"   CDC 端口：{p or '未找到（板子没插？）'}")
    except Exception as exc:                              # noqa: BLE001
        print(f"   ⛔ 端口探测失败：{exc!r}")

    if args.port or not args.dry_run:
        print("\n## 只读身份探测（**不发运动命令**）")
        try:
            from litearm_core import Arm
            from .capture import snapshot_identity
            arm = Arm(port=args.port).connect()
            try:
                ident = snapshot_identity(arm)
                for k in ("firmware", "n_joints", "ff_mask", "hold_kp_gain", "model"):
                    print(f"   {k}: {ident.get(k)}")
                st = arm.get_state(refresh=True).value
                if st is not None:
                    print(f"   flags=0x{st.flags:04X} flag_names={st.flag_names} "
                          f"joint_fault=0x{st.joint_fault:04X}")
            finally:
                arm.close()
        except Exception as exc:                          # noqa: BLE001
            print(f"   ⚠ 探测失败（不影响 doctor 结论）：{exc!r}")
    return 1 if bad else 0


def _print_card(card, *, quiet: bool) -> None:
    t = card.total
    if quiet:
        print("" if t is None else f"{t:.1f} {card.grade}")
        return
    print("\n" + "=" * 62)
    if t is None:
        print("总分：**无可计分维度** —— 不产生总分（未实测 ≠ 0 分）")
    else:
        print(f"总分：{t:.1f} / 100　等级 **{card.grade}**"
              f"　（{len(card.scored_dimensions)} 个已测维度加权）")
    print("=" * 62)
    for d in card.dimensions:
        s = "  —  " if d.score is None else f"{d.score:5.1f}"
        print(f"  {d.name:<14}{s}   ({d.measured_ratio} 项已计分)")
    if card.missing_dimensions:
        print("\n未参与总分：" + "、".join(card.missing_dimensions))
    miss = [m for d in card.dimensions for m in d.metrics if m.score is None]
    unt = [m for d in card.dimensions for m in d.metrics if not m.trusted]
    if unt:
        print(f"\n⛔ 不可信而剔除 {len(unt)} 项：" + "、".join(m.spec.key for m in unt))
    if miss:
        print(f"未实测 {len(miss)} 项：" + "、".join(m.spec.key for m in miss[:8])
              + ("…" if len(miss) > 8 else ""))


def _stamp() -> str:
    import time
    return time.strftime("%Y%m%d_%H%M")


def _registry():
    from .metrics import registry
    return registry


def _tiers():
    from .suites import tiers
    return tiers
