# EB-GRPO: 熵带闭环控制下的 GRPO 训练稳定性实验

大语言模型后训练强化学习（RLVR）中，策略熵会在训练过程中失控——先膨胀、后坍缩，并在坍缩后锁死奖励平台期。
本仓库是课程论文《大语言模型后训练强化学习中的熵稳定机制分析与算法设计》的**完整实验代码与结果**，包含：

- 一个零依赖的熵带控制器（死区 + 比例-积分 + 双执行器），以及 11 项单元测试；
- 把它外挂到 [verl](https://github.com/volcengine/verl) v0.8.0 的锚点式补丁（不改损失内核）；
- Countdown 任务的数据生成器（答案可程序校验）与 jsonl→parquet 转换；
- 3 组对照 × 2 随机种子 × 150 步的逐步骤日志、聚合表与绘图脚本。

## 主要结果（Qwen2.5-0.5B，150 步，2 种子均值）

| 组 | 带内步数比例 | 熵累积面积 | 末期熵 | 末期梯度范数 | 末期 KL | 末期奖励 |
|---|---|---|---|---|---|---|
| vanilla GRPO | 14% | 1.15 | 0.73 | 0.09 | 0.071 | 0.0499 |
| GRPO + 固定熵系数 0.001（开环） | 39% | 1.41 | 1.46 | 0.19 | 0.062 | 0.0498 |
| **EB-GRPO（闭环，本仓库）** | **50%** | **1.52** | **1.65** | **0.24** | **0.061** | 0.0496 |

“带内”指熵落在 `[0.6, 1.1]×H₀` 的比例（H₀ 为该 run 首步的 token 平均熵，即闭环组的控制目标）。

**读结果前请先看这条**：本设置下三组的组内平均奖励全部卡在 0.05（格式分上限），任务正确率实测为 0，
因此奖励维度没有区分度，`0.0496` 与 `0.0499` 的差异不支持任何性能结论。本仓库验证的是**熵动力学机制**，
不是基准性能。另外，闭环并非“稳稳守住”，而是呈“饱和抬升—过冲—反向惩罚”的极限环
（`results/README.md` 第 4 节有逐种子的量化说明）。

## 目录

```
experiment/
├── algo/entropy_controller.py      # 控制器（纯标准库）+ tests/（11 项单测）
├── algo/README_EB_GRPO.md          # 算法伪代码、verl 挂点、日志 schema
├── verl_integration/               # install_eb_patch.py（幂等补丁）、countdown_reward.py、run_eval.py
├── patches/eb_grpo_verl_v0.8.0.patch  # 打补丁后的完整 diff（供审阅/换版本重放）
├── data/make_countdown.py          # 数据生成 + 奖励函数（含自检）
├── data/convert_to_verl.py         # jsonl → verl parquet
├── scripts/{train,run_smoke,run_matrix,status}.sh
├── tools/{collect_metrics,summarize,plot_entropy,make_paper_figures}.py
└── env/{ENV_SETUP.md,setup_env.sh,freeze_*.txt}   # 版本锁定与踩坑记录
data/          # 8000 训练 / 512 测试（jsonl 与 parquet，seed=42）
results/       # 逐 run 日志（eb_log.jsonl + console.log）、主宽表、聚合表、索引 README
figures/       # 论文图 1–4（PDF + 600dpi PNG + 面板对齐/碰撞审计 JSON）
```

## 复现

```bash
# 1) 环境：版本组合是硬约束，务必按 experiment/env/ENV_SETUP.md 装
#    （verl v0.8.0 + vLLM 0.11.0 + torch 2.8.0 + tensordict 0.10.0 + flash-attn 2.8.3）
python experiment/verl_integration/install_eb_patch.py     # 把控制器装进 verl（幂等）

# 2) 数据已随仓库提供；如需重新生成：
python experiment/data/make_countdown.py --train-size 8000 --test-size 512 --seed 42 --out-dir data/countdown
python experiment/data/convert_to_verl.py --src data/countdown --out data/verl

# 3) 主对比（3 组 × 2 种子 × 150 步，单张 RTX 4090 约 12 小时）
bash experiment/scripts/run_matrix.sh

# 4) 汇总与出图
python experiment/tools/collect_metrics.py --dir runs --out results/main_metrics.csv
python experiment/tools/summarize.py --csv results/main_metrics.csv --out results/summary.csv
python experiment/tools/make_paper_figures.py
```

控制器单测：`python experiment/algo/tests/test_entropy_controller.py`（11 项，含两条闭环仿真）。

## 方法一句话

以每步训练日志里已有的 token 平均熵为观测量，经带死区的比例-积分律得到归一化控制量 u∈[−1,1]：
`u>0` 时提高熵系数（鼓励探索）并放宽非对称裁剪上界；`u<0` 时给负熵系数（抑制发散）。
熵带不写死绝对值，而取首个观测熵的固定比例，故同一组超参可跨模型规模复用。

## 数据与日志格式

`eb_log.jsonl` 每行一个优化步：`step, entropy, u, entropy_coeff, clip_higher, kl_loss, ppo_kl,
clipfrac, grad_norm`；闭环组另有 `error, integral, band_lo, band_hi, h_ref`。
`console.log` 是 verl 原始输出，含每步全部指标。二者由 `tools/collect_metrics.py` 按步号合并成
`results/main_metrics.csv`（1000 行 × 106 列），是论文图表的唯一数据源。

## 说明

- 实验为受控小规模机制演示（0.5B、单一任务、2 个种子），结论不外推到前沿规模；
  所有对比均为描述性统计，未做显著性检验。
- 训练配方与超参见 `experiment/scripts/train.sh` 与 `results/README.md` §1。
