"""PyQt5 界面 —— **只是同一套装配路径的另一个前端**。

⚠ 界面里没有任何一条"自己的采集/分析/安全判断"。它做的事只有三件：
挑动作表（`suites.tiers` / `suites.singles`）、调 `session.run_actions`、
把结果显示出来。安全闸（`safety.require` 与 `Executor._gate`）在底层，一视同仁。
"""
