"""版本号双写必须一致 —— 照 `litearm-server/tests/test_version_sync.py` 的惯例。"""
import re
from pathlib import Path


def test_pyproject_and_dunder_version_agree():
    root = Path(__file__).resolve().parent.parent
    m = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.M)
    assert m, "pyproject.toml 里找不到 version"
    from litearm_benchmark import __version__
    assert m.group(1) == __version__, (
        f"pyproject {m.group(1)!r} != __init__ {__version__!r} —— 改版本号要两处一起改")
