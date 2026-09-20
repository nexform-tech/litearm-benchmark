"""指标层：目录（阈值与分数曲线）、分析（原始值）、信号原语。

分层是刻意的 —— ``registry`` 只声明「什么算好」，``analysis`` 只算「实际是多少」，
``signals`` 只提供带教训的算子。打分（:mod:`..score`）在上一层，且是纯函数。
"""
from litearm_benchmark.metrics import analysis, registry, signals  # noqa: F401

__all__ = ["analysis", "registry", "signals"]
