# litearm-benchmark —— LiteArm 机械臂综合性能基准

它回答一个问题：**这台臂现在整体有多好？**

像跑分那样，把**位置跟踪精度、速度环跟踪、力矩、抖动、笛卡尔、多轴同步、热、
耐久、安全、固件计算性能、通信链路**这些维度各自打分，再按权重合成一个
**0~100 的总分与等级（S/A/B/C/D/F）**。建立在 [`litearm-core`](https://gitee.com/yudao_hz_1/litearm-core)
之上 —— 只走它的公开面（USB CDC 直连固件）。

```bash
pip install -e ".[dev]"

litearm-bench list                     # 看四个档位各跑什么
litearm-bench doctor                   # 环境与口径自检（不动机器人）
litearm-bench run --tier quick --go    # ~1 分钟（实测 58 s），会动真机
litearm-bench score --dir docs/bench_.../    # 改权重后重算，不动硬件
litearm-bench compare --a runA --b runB      # 两次运行逐指标对比
```

> 库用法见下方「快速开始」；API 面见 `src/litearm_benchmark/__init__.py` 的 `__all__`。

---

## 三条硬规矩

这三条是本工程与「随便量量」的分界线，都写在代码里、也都有测试守着：

1. **未实测 ≠ 0 分。** 测不出来（没接真机、没跑笛卡尔档、没装负载）就标「未实测」，
   **不参与**总分聚合，总分按实际测到的维度重新归一化。用 0 分把没测的项拖下去，
   是另一种撒谎。
2. **不可信 ≠ 高分。** 跨流（300Hz 参考 vs 100Hz 实测）算出来的数必须带**自证量**；
   自证不达标就**整组剔除**并写明原因。本项目历史上出过「对齐失效却照常出数」，
   那批 30~133 mrad 的记录全是伪影。
3. **仿真绝不出总分。** `--dry-run` 用 core 的桩传输，而**桩不建模运动** ⇒ `q` 恒定
   ⇒ 抖动 0 ⇒ 满分；`τ=0` ⇒ 力矩占比 0 ⇒ 满分。这种「什么都没测却得 78 分」是最危险的
   假绿，故在**聚合层**直接堵死，并在报告顶部挂红条。

## 档位

| 档位 | **实测挂钟** | 覆盖 | 人工 |
|---|---|---|---|
| `quick` | **~1 min** | 静差/重复性/跟踪/抖动/看门狗/固件自检 | 无 |
| `standard` | **~2.5 min** | 加笛卡尔、多轴同步、更多静止位形 | 无 |
| `full` | **~4 min** | 加长时间稳定性、热、多速度档 | 无 |
| `manual_load` | 视操作 | 装/拆配重前后的对照 | **需人工** |

> ⚠ 上表是 **2026-09-21 真机实测**（`quick` 58 s、`full` 4 min 11 s）。
> 早先写的 `~5 / ~15 / ~40 min` 是估的、从未核对，实际高了 6~10 倍。

动作**全是数据**（`suites/tiers.py`），执行器只有一份 —— 加指标时只加一条动作。

⚠ **运动包线**：全部动作落在**安全台架包线**内（`suites/base.py` 的 `POSES`）：
单轴行程 ≤ 0.8 rad、不碰 J4 的软限上界（它只有 **+0.0175 rad**，是全轴最危险的）、
不进 URDF 零位奇异。大范围/贴限位的能力**本工程不做**。

## 落盘的三层数据

寿命不同，所以分开存：

| 文件 | 内容 | 何时需要 |
|---|---|---|
| `trace.npz` | 原始双流 + TCP 采样 + 300Hz 日志窗口 | **最贵**（要动硬件）⇒ 一次存全，之后永不重采 |
| `raw.json` | 算出来的原始值 + 每条的证据/说明 + 机器身份 | 改**权重/阈值**时只需它（`score` / `compare`） |
| `REPORT.md` / `metrics.csv` | 人看的成绩单与明细 | 每次 rescore 重新生成，**不要手改** |

## 依赖关系

```
litearm-benchmark  ──依赖──>  litearm-core          （USB CDC 直连）
                   ──依赖──>  numpy                 （纯分析用）
                   ──不依赖──> pylitearm / pinocchio （刻意：不引自己的运动学模型）
```

取不到的量（母线电流、任意构型 FK、在途笛卡尔进度…）在 `instruments.UNAVAILABLE`
里**显式登记**，对应指标标「未实测」而不是静默返回 0。

## 测试

| 文件 | 测什么 |
|---|---|
| `tests/test_registry_and_score.py` | 指标曲线合规（锚点升序/分数界/单调方向）· 未实测≠0 · 自证闸 · 激励闸 · **负控：故意造坏曲线与假自证量，判据必须红** |
| `tests/test_signals_and_align.py` | 量化地板按**方差**扣除 · 巡航段取**最长连续**段（负控：末尾尖峰不许污染）· 抖动用**直线拟合残差** · seq 回绕与"可比对数" · 对齐找回已知时延 · **负控：两条无关的流必须判不可信** · δ 顶窗检测 |
| `tests/test_pipeline_offline.py` | 桩传输端到端（跑→分析→打分→落盘）· **仿真不出总分** · 重放与当时一致 · 含人工的档位拒绝非交互运行 |
| `tests/test_version_sync.py` | `pyproject.toml` 与 `__init__.py` 的版本号必须一致（家族惯例） |

```bash
pytest                     # 全离线（桩传输，不碰真机）
```

真机路径由 `litearm-bench run ... --go` 覆盖（本仓**不**在 pytest 里动真机）。

## 架构

```
src/litearm_benchmark/
  instruments.py    仪表边界：采样率/日志容量/各量化的 LSB/取不到的量 —— **常数唯一真相源**
  safety.py         动臂前的安全闸（限位裕度/J4 陷阱/锁存轴）+ 收尾（绝不 disable）
  session.py        会话与顶层编排：连臂 → 安全前置 → 采集 → 分析 → 打分 → 落盘
  capture.py        双流采集：100Hz 状态帧挂钩 + 300Hz 日志分窗（≤2400 拍/窗）
  align.py          跨流对齐：互相关 + **自证量**（错开 80ms 的对照）+ 顶窗检测
  metrics/
    registry.py     指标目录：38 项指标的阈值曲线与**出处**（实测/物理下界/标称/占位）
    signals.py      信号原语：稳健标尺/巡航段/直线拟合残差/seq 对账 —— 每个都带一条历史教训
    analysis.py     从 Trace 算**原始值**（从不算分）
  score.py          原始值 → 指标分 → 维度分 → 总分/等级（纯函数）
  report.py         Markdown 成绩单 + CSV + 三层落盘
  replay.py         离线重放与两次运行对比（不动硬件）
  suites/           档位（动作是数据）与执行器
  cli.py            命令行（argparse，家族风格）
```

## 相关项目

- [`litearm-core`](https://gitee.com/yudao_hz_1/litearm-core) —— 本工程依赖的直连 SDK
- [`litearm-server`](https://gitee.com/yudao_hz_1/litearm-server) —— zenoh/WebSocket 服务层
- [`litearm-stm32`](https://gitee.com/yudao_hz_1/litearm-stm32) —— 固件

## 许可证

MIT
