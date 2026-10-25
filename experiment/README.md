# experiment/ — 论文实验代码包

《大语言模型后训练强化学习中的熵稳定机制分析与算法设计》实验部分。
目标：**单卡 4090 约一个工作日**完成"复现熵坍缩 + 验证 EB-GRPO 闭环控制"。

```
experiment/
├── README.md                 <- 本文件（总览 + GPU 需求清单）
├── env/ENV_SETUP.md          <- 环境搭建与版本锁定表（W1 执行）
├── data/make_countdown.py    <- Countdown 数据生成器 + 奖励验证器（纯标准库，已自检）
├── algo/
│   ├── entropy_controller.py <- EB-GRPO 控制器核心（纯标准库，已单测）
│   ├── README_EB_GRPO.md     <- verl 集成说明 + 日志 schema（挂点待 W1 核对）
│   └── tests/                <- 控制器单元测试（7 项，含闭环有界性仿真）
├── scripts/train.sh          <- 三组对比实验统一启动骨架（GROUP/SEED 驱动）
└── tools/plot_entropy.py     <- 熵轨迹论文图工具（灰阶 300dpi，已自检）
```

## 三组实验

| 组 | entropy_coeff | clip_higher | 说明 |
|---|---|---|---|
| baseline | 0（恒定） | 0.2 | vanilla GRPO，预期熵坍缩 |
| fixed | 固定 c（W1 定） | 0.2 | 开环基线 |
| eb | 控制器调节 | 控制器调节 | EB-GRPO（本文） |

每组 2 seeds × 100–150 步，模型 Qwen2.5-1.5B（0.5B 用于预跑调参），
任务 Countdown（`data/make_countdown.py` 程序化生成，答案可验证）。

## 本地（无 GPU）已验证的部分

- `make_countdown.py --selftest`：200 题全部有解可验证，6 类干扰表达式全部拒绝，奖励函数 4 项断言通过
- `tests/test_entropy_controller.py`：饱和/带内死区/线性区/抗积分饱和/配置校验/闭环有界性，7 项通过
- `plot_entropy.py --selftest`：合成数据出图通过

## GPU 到位后（按 env/ENV_SETUP.md §4 冒烟清单执行）

1. 环境安装与版本锁定 → `pip freeze` 留痕
2. 模型下载、jsonl→parquet 转换（字段按锁定 verl 版本定）
3. 0.5B 预跑 100 步：**确认 vanilla GRPO 熵坍缩可复现**（实验成立前提）
4. 校准熵带 [h_lo, h_hi] 与 fixed 组系数 c
5. 三组 × 2 seeds 正式跑（`GROUP=xxx SEED=n bash scripts/train.sh`）

## 向用户申请 GPU 的标准话术（何时说、说什么）

- **第一次（W1 预跑）**：1×4090（24GB）约 2 小时——装环境 + 0.5B 预跑
- **第二次（W2 主实验）**：同一张卡连续约 1 个工作日（或 1×A100 半天）
- 服务器要求：Linux、CUDA ≥ 12、磁盘 ≥ 200GB、能访问 HuggingFace（或提供镜像）
