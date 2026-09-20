"""套件层：动作是数据（:mod:`tiers`），执行器只有一份（:mod:`base`）。"""
from litearm_benchmark.suites import base, tiers  # noqa: F401

__all__ = ["base", "tiers"]
