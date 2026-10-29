# litearm-benchmark

LiteArm 七轴机械臂的电流环力矩估计精度评估工具：从机械臂采集关节状态与电流环
力矩，回放刚体逆动力学，一条命令得到逐轴误差、图表与误差来源归因。

[English](README.md)

## 范围

- 关节级采集，最高 100 Hz：位置、速度、电流环力矩、MOS/绕组温度、逐轴故障码。
- 基于固件在线刚体模型（RNEA）的逆动力学力矩计算。
- 逐轴误差指标（RMSE / MAE / bias / peak / R² / 相关系数）、图表、延迟扫描、
  按证据排序的误差来源诊断。
- 三个激励场景：静态持位、正弦扫掠、慢速多姿态。

工具输出（命令行文本、报告、图）为中文；本 README 与 `docs/` 文档为英文。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate

# 离线分析与测试：不需要硬件, 也没有第三方运行时依赖。
pip install -e ".[test]"

# 只有真机采集才需要: 额外安装可选的 litearm SDK。
pip install -e ".[sdk]"
```

SDK 是可选依赖且延迟导入，因此离线分析和测试无需安装。未安装时，采集命令会
直接提示 `pip install -e ".[sdk]"` 并指向离线的 `analyze` 路径。

## 快速开始

```bash
# 只打印采集计划和当前位形, 不运动。
litearm-tau

# 真正采集: hold + sine + poses, 逐个场景分析与出图出报告。
litearm-tau --go

# 只采集一个场景。
litearm-tau collect --scenario sine --go

# 对已有会话目录或包含多个场景的 run 目录做离线分析。
litearm-tau analyze runs/20261029_120000_tau_eval --outdir /tmp/tau
```

未安装成脚本时，`python3 -m litearm_benchmark.torque_eval ...` 与
`litearm-tau` 等价。

常用参数：

| 参数 | 含义 |
|---|---|
| `--go` | 真正 enable、运动、采集。不加则只读。 |
| `--scenario all\|hold\|sine\|poses` | 要跑的场景（默认 `all`）。 |
| `--fs` | 落盘采样率, 1..100 Hz（默认 100）。 |
| `--amp`、`--freq` | 正弦幅值（rad）与频率（Hz）；幅值会按软限位余量自动收窄。 |
| `--sine-joints 1,2,3` | 只激励指定关节的正弦段（1 基）。 |
| `--gravity x,y,z` | 分析用重力向量（默认 `0,0,-9.81`）。台架重力沿 +Y 时用 `0,9.81,0`。 |
| `--cutoff-dq`、`--cutoff-ddq`、`--cutoff-tau` | 零相位低通截止频率（Hz）。 |
| `--ambient-c` | 手工记录的室温；固件只报绕组/MOS 温度。 |
| `--no-plot`、`--no-report` | 跳过 SVG 图或 Markdown 报告。 |

## 场景

- **hold** —— 在当前位形静态持位（约 3 s）。用于隔离重力模型与电流环零偏；
  此时 `ddq = dq = 0`, 残差就是静态 bias。
- **sine** —— 连续伺服正弦（约 10 s, 0.2 Hz）, 指令重发率 50 Hz。激励惯量、
  科氏与摩擦项；幅值自动收窄，保证每个选中关节保留 0.15 rad 软限位余量。
- **poses** —— 慢速多姿态序列（4 个目标位形, 每姿态停驻 1.2 s），同样自动收窄
  幅度。可暴露随方向变化的摩擦误差与不同构型下的重力误差。

所有规划目标在执行前都要通过软限位余量检查。

## 采集数据

一个会话目录包含：

- `status.csv` —— 每个均匀采样一行：`t, seq, mode, mode_name, flags,
  joint_fault`, 然后是每个关节的 `q{i}, dq{i}, tau{i}, tmos{i}, tcoil{i},
  err{i}`。
- `meta.json` —— 场景描述、采样率、样本数、时间跨度、室温（若提供）及其他
  运行元数据。
- `model.json` —— 从机械臂读回的固件模型快照：刚体参数、电机惯量 `jm`、
  payload、重力、固件版本与几何指纹。

到达间隔不均匀时，分析前会线性重采样到均匀网格，报告里会明确标注。

## 处理链

1. 按记录的采样率把 `status.csv` 重采样到均匀网格。
2. 零相位 Butterworth 滤波（`filtfilt`）：`q`/`dq` 15 Hz, 实测 `tau` 10 Hz；
   `ddq` 由滤波后的 `dq` 中心差分得到，再 5 Hz 低通。
3. 两端裁剪（`max(2/cutoff_ddq, 0.1) s`, 上限为 span/5），避开零相位滤波的
   两端瞬态。
4. 回放世界系牛顿-欧拉 RNEA（固件 `rnea_core_m` 的逐行移植）：
   `tau = M·ddq + C·dq + G + jm·ddq + friction`。
5. 残差 `tau_meas − tau_model`, 关节侧, N·m；`scaled` 口径按固件
   `gravity_scale`/`inertia_scale` 换算。
6. 计算指标、按速度分层、延迟扫描、残差相关性，然后出图与 Markdown 报告。

## 输出

分析结果写入会话目录（或指定的 `--outdir`）：

- `timeseries.csv` —— 滤波后的 `q, dq, ddq`、实测/模型/换算力矩，以及每个
  关节的 G/C/I/friction 各项。
- `metrics.json` —— 汇总与逐轴指标、速度分层、延迟扫描、残差相关性、各项
  RMS、诊断列表。
- `figures/*.svg` —— 33 张纯标准库生成的 SVG：逐轴力矩与残差曲线、RMSE/bias
  柱状图、parity、延迟扫描、残差直方图。
- `report.md` —— 中文 Markdown 报告：指标表、方法与口径、诊断、图索引、限制。
- 多场景 run 目录还会生成 `summary.md` 汇总对比所有场景。

## 诊断

报告按证据强度排序列出误差来源，而不是给出最终判决。典型信号：

| 信号 | 证据 | 先查什么 |
|---|---|---|
| 模型与实测高度一致 | 平均相对 RMSE < 2 % | 采集与 RNEA 链路；合成数据下这是预期结果 |
| 时序/滤波延迟 | 最优时移可明显降低 RMSE | `cutoff_ddq` 是否过低，或电流环报数延迟 |
| 摩擦/低速换向 | 残差与摩擦项相关 | Stribeck 参数；看 hold/低速段与换向点 |
| 重力模型/电流环零偏 | `ddq = dq = 0` 时仍有静止残差 | 台架真实重力向量（`--gravity`）、`get_ff_scalar(6)`、电流环标定 |
| 惯量/加速度项 | 残差与惯量项相关 | `jm`/惯量缩放；对比 sine 段 `ddq` 幅值与残差符号 |
| 未见单一主导项 | 误差混合 | 逐轴 `metrics.json` 与残差分布 |

每一条在 `report.md` 里都附有具体建议。

## 参考示例

`examples/make_synthetic.py` 用出厂模型自身合成一段确定性会话，残差只有数值
噪声，可在无硬件条件下跑通完整的 采集 → 分析 → 出图 → 报告 链路：

```bash
python3 examples/make_synthetic.py --outdir runs/synthetic_sine
```

提交在仓库里的参考输出位于
[`examples/reference_sine/`](examples/reference_sine/)（报告：
[report.md](examples/reference_sine/report.md)）。日常输出写到被 git 忽略的
`runs/` 目录。提交的参考输出是一个小而确定性的例外；请重新生成它，而不是
原地分析，因为分析会覆盖目标目录里的文件。

## 安全

- 命令默认只读：只打印计划和当前位形。任何 enable、运动或采集都必须加
  `--go`。
- 所有规划目标执行前都过软限位检查，留 0.15 rad 余量；幅度会自动收窄，若没有
  安全幅度则规划直接失败。
- 采集结束刻意**不** disable（在该位形失能会因自重坠回）。急停请放手边。
- 机械臂有故障位时拒绝执行采集。

## 限制

- `tau` 是**电流环估计值**，不是关节力矩传感器；电流环标定误差会直接进入结论。
- 固件模型是辨识回归量：link mass 多为 1、惯量含负值、`M` 不正定。只用于
  前向计算，**不可求 M⁻¹**（仿真或控制）。
- 静态重力核查只在末段静止时有效；运动段不冒充静态。
- 固件读不到环境温度；需要时用 `--ambient-c` 手工记录。
- 结论的适用范围是本组构型与速度；外推到限位附近或高速段需另采数据。

## 测试

```bash
pip install -e ".[test]"
pytest
```

测试完全离线：不需要硬件，也不需要 NumPy/Matplotlib（图为纯 SVG）。源码
checkout 未安装时，用 pytest 7 及以上配合 `PYTHONPATH=src pytest`。

## 文档

- [文档索引](docs/index.md)
- [安装与验证](docs/setup.md)
- [架构](docs/architecture.md)

## 许可证

MIT
