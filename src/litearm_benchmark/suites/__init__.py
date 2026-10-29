"""套件层：动作是数据（:mod:`tiers` 档位 / :mod:`singles` 单项），执行器只有一份（:mod:`base`）。"""
from litearm_benchmark.suites import base, singles, tiers  # noqa: F401

__all__ = ["base", "singles", "tiers"]
