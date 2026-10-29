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


def _fake_port(dev, vid=None, pid=None, product=None):
    """造一个 pyserial `ListPortInfo` 的替身（只带被测代码真正读的那几个字段）。"""
    return type("P", (), {"device": dev, "vid": vid, "pid": pid, "product": product})()


#: 真实布局：31 个主板串口在前、臂在中段（第 32 位）、后面还跟着一台 USB 调试器。
#: ⚠ `ttyACM1` 那台 ATK CMSIS-DAP 是**调试器不是臂** —— 按编号猜一定会认错。
_TTYS = [_fake_port(f"/dev/ttyS{i}") for i in range(31)]
_LAYOUT = _TTYS + [_fake_port("/dev/ttyACM0", 0x1D50, 0x606F, "STM32 Virtual ComPort"),
                   _fake_port("/dev/ttyACM1", 0x04D8, 0x00DF, "ATK-FS-HID-CMSIS-DAP")]


def test_arm_is_identified_and_sorted_first(monkeypatch):
    """臂必须被**认出来**（打 ★）并排在第一 —— 不能在 34 条里靠人找。

    ⚠ 守的是一个真实踩过的坑：pyserial 会把 `/dev/ttyACM0` **也列进去**（本机第 32 位）。
    所以"插到最前"不能写成 `if p not in out: out.insert(0, p)` —— 那个条件恰好恒假。
    """
    import serial.tools.list_ports as lp
    import litearm_core

    from litearm_benchmark.gui.main_window import _list_ports

    monkeypatch.setattr(lp, "comports", lambda: list(_LAYOUT))
    monkeypatch.setattr(litearm_core, "find_cdc_port", lambda: "/dev/ttyACM0")
    infos = _list_ports()

    assert infos[0].device == "/dev/ttyACM0", "臂没排到第一位 —— 下拉会默认选中主板串口"
    assert infos[0].is_arm and "★" in infos[0].label and "1d50:606f" in infos[0].label, \
        "认出来了却没标出来 —— 界面上看不出哪个是它"
    # 负控：**只**认那一台。调试器在同一条 USB 总线上、也是 ttyACM，不许被认成臂。
    assert [i.device for i in infos if i.is_arm] == ["/dev/ttyACM0"], \
        "把别的 USB 串口也认成臂了"
    assert infos[0].is_usb, "臂是 USB 设备 ⇒ 默认过滤也必须留着它"
    # 非 USB 串口仍**在结果里**（过滤是界面的事，取数函数不替调用方决定什么是噪声）
    assert sum(1 for i in infos if not i.is_usb) == 31


def test_unidentified_arm_still_lists_everything(monkeypatch):
    """**负控**：认不出来时不许抛异常、不许吞掉清单，只是没有 ★。"""
    import serial.tools.list_ports as lp
    import litearm_core

    from litearm_benchmark.gui.main_window import _list_ports

    monkeypatch.setattr(lp, "comports", lambda: list(_LAYOUT))
    monkeypatch.setattr(litearm_core, "find_cdc_port", lambda: None)
    infos = _list_ports()
    assert len(infos) == 33, "找不到臂时清单被改动了"
    assert not any(i.is_arm for i in infos) and not any("★" in i.label for i in infos)


def test_arm_found_but_not_listed_by_pyserial_is_still_added(monkeypatch):
    """认出来了、pyserial 却没列 ⇒ **必须补一条**，否则界面上根本选不到它。"""
    import serial.tools.list_ports as lp
    import litearm_core

    from litearm_benchmark.gui.main_window import _list_ports

    monkeypatch.setattr(lp, "comports", lambda: [_fake_port("/dev/ttyS4")])
    monkeypatch.setattr(litearm_core, "find_cdc_port", lambda: "/dev/ttyACM9")
    infos = _list_ports()
    assert infos[0].device == "/dev/ttyACM9" and infos[0].is_arm, "认出来了却选不到"


def test_non_usb_ports_are_hidden_by_default_but_reachable(win, monkeypatch):
    """界面默认**只列 USB 串口**，勾上就全回来 —— 过滤，不是删除。"""
    import serial.tools.list_ports as lp
    import litearm_core

    monkeypatch.setattr(lp, "comports", lambda: list(_LAYOUT))
    monkeypatch.setattr(litearm_core, "find_cdc_port", lambda: "/dev/ttyACM0")

    win._refresh_ports()                                # noqa: SLF001 - 就是被测的刷新
    assert not win.chk_all.isChecked(), "默认应是隐藏（本机 34 个里 31 个是主板串口）"
    assert win.port.count() == 2, "默认该只列两台 USB 设备"
    assert [win.port.itemData(i) for i in range(win.port.count())] == \
        ["/dev/ttyACM0", "/dev/ttyACM1"], "顺序或取回的设备路径不对"
    assert win.port.currentData() == "/dev/ttyACM0", "默认选中的不是臂"
    assert "已识别 LiteArm" in win.lbl_note.text(), "没告诉操作者识别结果"

    win.chk_all.setChecked(True)
    assert win.port.count() == 33, "勾上后非 USB 串口没回来"
    assert win.port.currentData() == "/dev/ttyACM0", "过滤切换后把选中的臂弄丢了"


def test_selection_survives_a_refresh(win, monkeypatch):
    """手选过某台设备后，刷新**不许**把选择弄丢。

    ⚠ 回选按 `currentData()`（设备路径）而不是显示文字：标签里带 ★ 与 VID:PID，
    按文字比对一旦臂插拔后 ID 变了，就会**静默丢选**、退回第一条。
    """
    import serial.tools.list_ports as lp
    import litearm_core

    monkeypatch.setattr(lp, "comports", lambda: list(_LAYOUT))
    monkeypatch.setattr(litearm_core, "find_cdc_port", lambda: "/dev/ttyACM0")

    win._refresh_ports()                                # noqa: SLF001
    win.port.setCurrentIndex(win.port.findData("/dev/ttyACM1"))
    win._refresh_ports()                                # noqa: SLF001
    assert win.port.currentData() == "/dev/ttyACM1", "刷新把操作者手选的端口弄丢了"


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
