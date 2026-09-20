"""离线测试的公共夹具。

照本家族的惯例：**自带 sys.path 注入（免安装）**，并提供造臂处 —— 真机与离线
走同一条装配路径（`Arm(transport_factory=...)`）。
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def fake_factory():
    from litearm_core.testing import FakeTransport

    def make(port="fake", **kw):
        return lambda p: FakeTransport(p or port, **kw)
    return make


@pytest.fixture
def offline_arm(fake_factory):
    from litearm_core import Arm
    arm = Arm(port="fake", transport_factory=fake_factory()).connect()
    yield arm
    arm.close()
