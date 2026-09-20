"""界面冒烟 —— **无头**跑（`QT_QPA_PLATFORM=offscreen`），不需要显示器。

界面最容易出的两类错：**导入期/构造期就炸**、以及**把清单渲染漏了**。
这两类都在这里拦住；真正需要人眼的交互不测（那是手工的事）。

⚠ 这里**不碰真机** —— 工作线程在没有端口时 `_idle` 直接返回。
⚠ 关窗时走 `worker.shutdown()`，这与真实关窗同一条路径（`closeEvent`）。
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication

    from litearm_benchmark.gui.app import _install_excepthook
    # ⚠ 必须装：PyQt5 在**槽里**抛出的未捕获异常会 `qFatal` ⇒ 测试进程**核心转储**，
    #   而那样 pytest 连是哪个断言失败都报不出来（只给一个 traceback 到 libc）。
    #   装了钩子之后，异常变成一条日志 + 测试正常失败。
    _install_excepthook()
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def win(qapp):
    from litearm_benchmark.gui.main_window import MainWindow
    w = MainWindow()
    yield w
    w.worker.shutdown()
    w.close()


def test_window_builds_and_lists_every_item(win):
    """两个清单必须与动作表**逐项对齐** —— 漏一项就是界面上少一个测试项。"""
    from litearm_benchmark.suites import singles, tiers

    assert win.lst_tiers.count() == len(tiers.TIERS), "档位清单漏项"
    assert win.lst_singles.count() == len(singles.SINGLES), "单项清单漏项"
    # 每一项都要能取回它的 key（拿不到 UserRole 的话按钮点了也没用）
    for i in range(win.lst_singles.count()):
        assert win.lst_singles.item(i).data(256), f"第 {i} 项没有 key"
    assert win.card_view is not None and win.log is not None


def test_tier_page_matches_the_action_table(win):
    """切到某一档时，描述要如实反映它的动作数与"含人工步骤"。"""
    from litearm_benchmark.suites import tiers
    for i, key in enumerate(tiers.TIERS):
        win.lst_tiers.setCurrentRow(i)
        txt = win.desc_tier.text()
        assert key in txt
        manual = any(a.is_manual for a in tiers.actions_of(key))
        assert ("含人工步骤" in txt) == manual, f"{key} 的人工提示与实际不符"


def test_card_view_flags_simulated_data_and_never_shows_a_total(qapp):
    """**负控**：桩数据在界面上必须挂红条、且**不显示总分**。

    这条守的是本工程的立身之本 —— 「仿真绝不出总分」。界面若把它渲染成一个
    正常的总分，命令行挡住的假绿就从窗口漏出来了。
    """
    from litearm_benchmark.gui.widgets import CardView
    from litearm_benchmark.session import run

    res = run("quick", dry_run=True, outdir=None, interactive=False)
    v = CardView()
    v.set_card(res.card, raw=res.raw, title="桩")
    banner = v._banner.text()                       # noqa: SLF001 - 就是被测的渲染
    assert "仿真" in banner and "不给仿真出总分" in banner
    assert res.card.total is None


def test_card_view_renders_untested_with_reasons(qapp):
    """**未实测必须带理由**显示 —— 界面上"未实测"与"得了 0 分"长得一样的话，
    三条硬规矩的第一条就白写了。"""
    from litearm_benchmark.gui.widgets import CardView
    from litearm_benchmark.session import run

    res = run("quick", dry_run=True, outdir=None, interactive=False)
    v = CardView()
    v.set_card(res.card, raw=res.raw, title="桩")
    txt = v._skipped.toPlainText()                  # noqa: SLF001
    assert "未实测" in txt
    # 未实测的项要带得出理由（`skip()` 写的 note），不能只有光秃秃一个键名
    assert "没有记录理由" not in txt, "有未实测项没写明理由"
