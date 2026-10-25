# results/ — 实验数据与结果索引

《大语言模型后训练强化学习中的熵稳定机制分析与算法设计》的实测产物。
本目录与 `experiment/` 一起构成可复现包：**代码 + 数据 + 逐过程日志 + 聚合表**。

## 1. 实验一览

| 项 | 取值 |
|---|---|
| 任务 | Countdown 数字游戏（程序化生成，答案可验证；奖励 = 格式分 0.05 + 正确 1.0） |
| 模型 | Qwen2.5-0.5B（base，非 instruct） |
| 算法 | GRPO（Double-clip 关，token-mean 损失，G=8，批 32，≤512 token，lr=1e-6，KL loss coef=1e-3） |
| 训练量 | 每组 150 步 × 2 种子（0/1），共 6 次；另有一次 100 步预跑用于标定熵带 |
| 探索策略三组 | `baseline`（entropy_coeff≡0）/ `fixed`（≡0.001 开环）/ `eb`（本文 EB-GRPO 闭环） |
| 硬件与软件 | 1× RTX 4090 24GB；verl v0.8.0 + vLLM 0.11.0 + torch 2.8.0 + flash-attn 2.8.3，详见 `experiment/env/ENV_SETUP.md` |
| 墙钟 | 42–51 s/步；6 次共 12 小时（2026-10-24 21:15 → 10-25 09:15） |

## 2. 目录结构

```
results/
├── README.md              <- 本文件
├── main_metrics.csv       <- 宽表：1000 行 × 106 列，逐步骤全指标（论文图表的唯一数据源）
├── summary.csv            <- 聚合表：每个 run 一行的汇总指标
├── prerun_baseline_05b_s0/  <- 预跑（100 步，用于标定熵带与观察现象）
└── runs/
    ├── baseline_seed0/  ├── baseline_seed1/
    ├── fixed_seed0/     ├── fixed_seed1/
    └── eb_seed0/        └── eb_seed1/
```

每个 run 目录内：

| 文件 | 内容 |
|---|---|
| `eb_log.jsonl` | 逐 optimizer step 一行：`step, entropy, u, entropy_coeff, clip_higher, kl_loss, ppo_kl, clipfrac, pg_loss, grad_norm`；EB 组另有 `error, integral, band_lo, band_hi, h_ref`（熵带自标定结果） |
| `console.log` | verl 原始输出，含每步 `step:N - k:v - …` 全指标行（奖励、响应长度、耗时、显存） |

`main_metrics.csv` 由 `experiment/tools/collect_metrics.py` 把上述两者按步号合并而成。

## 3. 主要结果（150 步，2 种子均值）

评价口径：参考熵 `H0` = 该 run 第 1 步的 token 平均熵；**评价带 = [0.6·H0, 1.1·H0]**
（即 EB 组的控制目标，同样作为另两组"是否守住探索水平"的标尺，三组才可比）。

| 组 | 末期熵（末20步均值） | 全程最低熵 | 熵 AUC | 带内比例 | score（末20） | KL | grad_norm（末20） |
|---|---|---|---|---|---|---|---|
| baseline | 0.731 | 0.544 | 1.153 | 14% | 0.0499 | 0.0712 | 0.091 |
| fixed | 1.460 | 0.789 | 1.405 | 39% | 0.0498 | 0.0616 | 0.190 |
| **eb（本文）** | **1.653** | 0.627 | **1.520** | **50%** | 0.0496 | **0.0609** | **0.238** |

逐种子（离散度，n=2 只作描述性报告，不做显著性主张）：

| run | 末期熵 | 最低熵 | 熵 AUC | 带内 | score | 控制量动作占比 | u 饱和占比 |
|---|---|---|---|---|---|---|---|
| baseline_seed0 | 0.431 | 0.339 | 1.018 | 10% | 0.0499 | — | — |
| baseline_seed1 | 1.030 | 0.749 | 1.287 | 19% | 0.0498 | — | — |
| fixed_seed0 | 1.636 | 0.888 | 1.499 | 51% | 0.0498 | — | — |
| fixed_seed1 | 1.285 | 0.690 | 1.311 | 28% | 0.0497 | — | — |
| eb_seed0 | 1.399 | 0.699 | 1.479 | 58% | 0.0500 | 93% | 21% |
| eb_seed1 | 1.908 | 0.556 | 1.561 | 41% | 0.0493 | 93% | 49% |

## 4. 读结果时必须知道的三件事

1. **score 没有区分度**：三组的 `critic/score/max` 恒为 0.05，即全部得分都来自格式分，
   Countdown 的任务正确率实测为 **0**。因此"测试通过率不劣"在本设置下只是
   弱检验（所有组都卡在同一个格式分上限），不能当作性能主张；本实验的定位是
   **熵动力学机制演示**，不是基准比较。
2. **开环基线并不弱**：固定 0.001 熵系数已能把带内比例从 14% 提到 39%。
   EB 的增量是再提到 50%、熵 AUC 再高 0.115、末期梯度范数高 25%，
   并且**不需要预先知道该用多大的系数**（执行器幅度由反馈决定）。
3. **闭环呈极限环振荡，不是稳稳守住带**：EB 组 93% 的步数在动作，其中 21%（seed0）/
   49%（seed1）打在 `±coeff_max` 边界上。seed1 的轨迹最能说明问题：38 步 H=1.17（带下界），
   45–50 步跌到 0.59 并拉满 `+0.01`，约 35 步后（80 步 H=1.79）回到带内，却在 90 步
   **过冲到 2.47（超出带上界 2.15）**，控制器随即转为惩罚，110–129 步又跌回带下方
   （全程 53/150 步低于带下界，最后一次是 129 步），150 步再次过冲到 2.42。
   即"饱和抬升 → 过冲 → 反向惩罚 → 再回落"的极限环：固定增益 `k_p=4` 对这个
   增益随训练阶段剧烈变化的对象偏大，且一步反馈延迟进一步削弱稳定裕度。
   这是本方法的主要局限，也是最有价值的可分析现象（论文第 5.3、6 章）。

## 5. 复现

```bash
# 1) 环境（版本锁定见 experiment/env/ENV_SETUP.md）
bash experiment/env/setup_env.sh   # 注意：需按 ENV_SETUP.md §1b/§2 修正 vLLM/tensordict/flash-attn 版本

# 2) 数据
python experiment/data/make_countdown.py --train-size 8000 --test-size 512 --seed 42 --out-dir data/countdown
python experiment/data/convert_to_verl.py --src data/countdown --out data/verl

# 3) 把 EB 控制器装进 verl（幂等，改动存档在 experiment/patches/）
python experiment/verl_integration/install_eb_patch.py

# 4) 主对比（3 组 × 2 种子 × 150 步，单卡约 12 小时）
bash experiment/scripts/run_matrix.sh

# 5) 汇总
python experiment/tools/collect_metrics.py --dir runs --out results/main_metrics.csv
python experiment/tools/summarize.py --csv results/main_metrics.csv --out results/summary.csv
```

控制器本身的单元测试：`python experiment/algo/tests/test_entropy_controller.py`（11 项）。

## 6. 数据契约

`eb_log.jsonl` 的字段语义、为什么熵是正值、为什么有一步反馈延迟，见
`experiment/algo/README_EB_GRPO.md` §3。`prerun_baseline_05b_s0/console.log` 没有
`RUN_EXIT` 标记，是因为它在写出 100 步日志后被人工终止（用于标定，非正式对比组），
数据本身完整可用。
