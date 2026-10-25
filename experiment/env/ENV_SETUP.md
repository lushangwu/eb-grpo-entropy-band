# GPU 服务器环境搭建（W1 已执行，2026-10-24 实测留痕）

> 原则：**版本一次锁定并留痕**。下表是智算中心实例上实际跑通的版本，完整快照见
> `freeze_20261024_v2.txt`（升级后）与 `freeze_20261024.txt`（初次搭建）。
> `setup_env.sh` 里最初的 torch/vLLM 组合已被实测否定，须按 §1b 修正才能起训。

## 0. 硬件与实例

- 实例：深圳大学智算中心 1× RTX 4090（24GB），Ubuntu 22.04，驱动 575.64.05，
  CUDA 工具链 12.1（`/usr/local/cuda-12.1`），14 vCPU，~1TB 内存，磁盘 4.6TB 可用。
- 访问：`~/.ssh/config` 中的 `智算中心_强化学习`（SOCKS5 代理 + 密钥）。
  长任务一律挂 `tmux`——ssh 断开时后台 `&` 进程会被带走。
- 出网：`huggingface.co`、`pypi.org` 直连不通；走 `HF_ENDPOINT=https://hf-mirror.com`
  与清华 PyPI 镜像（`/etc/pip.conf` 已配）。GitHub API 与 release 资源可下。
  实测带宽约 1.5–1.7 MB/s 且**多路并发不分摊总带宽**，装一次 vLLM 约 40 分钟。

## 1. 版本锁定表（2026-10-24 实测跑通）

| 组件 | 锁定值 | 依据 |
|---|---|---|
| Python | 3.10（conda env `ebgrpo`，`/opt/conda/envs/ebgrpo`） | verl 支持区间 |
| PyTorch | 2.8.0+cu128 | 由 vLLM 0.11.0 决定（0.8.4 时代是 2.6.0+cu124） |
| vLLM | **0.11.0** | verl v0.8.0 的 async rollout 要 `vllm.entrypoints.cli.serve.run_headless`；其 `setup.py` 声明 `vllm>=0.8.5,<=0.12.0`，源码另带 0.11–0.14 兼容分支 |
| verl | v0.8.0（tag，commit `7aed6b2`），`pip install --no-deps -e .` | GRPO 原生 `entropy_coeff` / `clip_ratio_low/high` |
| tensordict | **0.10.0** | 0.8.0 时 `DataProto→TensorDict` 直接报 "requires tensordict version 0.10" |
| transformers / tokenizers | 4.57.6 / 0.22.2 | 随 vLLM 0.11 一起升级 |
| flash-attn | 2.8.3.post2（GitHub release 预编译轮 `cu12torch2.8cxx11abiTRUE-cp310`） | CUDA 路径下 verl 用 `flash_attn.bert_padding` 做变长打包，是**硬依赖**（不是可选加速）；4090 上前向已单测 |
| ray | 2.59.0 | 随 vLLM 0.11 升级 |

## 1b. 实测踩到的五条约束（别再重复摸索）

1. **verl v0.8.0 删掉了 sync rollout**（`Rollout mode 'sync' has been removed`），
   单卡也只能走 async + vLLM server，于是 vLLM 版本被 verl 的版本区间锁死。
2. **`attn_implementation` 不是模型配置的直接键**：要写
   `+actor_rollout_ref.model.override_config.attn_implementation=...`
   （`override_config` 是 struct dict，不加 `+` 会报 "Key ... is not in struct"）。
3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 与 vLLM 的 CuMemAllocator
   （sleep 模式显存池）互斥**，vLLM worker 启动即断言失败——脚本里不要用。
4. **verl 的 `BaseConfig` 在 `__init__` 后冻结字段**：控制器逐步改写
   `entropy_coeff` / `clip_ratio_high` 必须登记进 `ActorConfig._mutable_fields`
   （verl 官方可变通道），否则 `FrozenInstanceError`。已由 `install_eb_patch.py` 完成。
5. **单卡共卡的显存瓶颈是词表 logits，不是参数**：微批按"条数"分时，
   8 条 ×768 token × 15.2 万词表在反向要 2.93 GiB 单次分配，24GB 直接 OOM。
   改为按 token 数动态分批（`use_dynamic_bsz=True` + `ppo_max_token_len_per_gpu=2048`）
   并开分块熵之后，0.5B 峰值降到 9.9GB（分配）/15.9GB（预留）。

## 2. 安装序列（实际执行版）

```bash
conda create -n ebgrpo python=3.10 -y && conda activate ebgrpo
pip install torch==2.8.0 vllm==0.11.0 "ray[default]==2.59.0" \
  transformers==4.57.6 "tensordict==0.10.0" numpy==1.26.4
git clone --depth 1 --branch v0.8.0 https://github.com/verl-project/verl.git
cd verl && pip install --no-deps -e .
pip install accelerate==1.6.0 codetiming datasets==3.5.0 dill hydra-core==1.3.2 \
  pandas==2.2.3 pyarrow==19.0.0 peft==0.15.1 pylatexenc torchdata==0.10.0 \
  wandb packaging uvicorn fastapi latex2sympy2_extended math_verify tensorboard
# flash-attn 走预编译轮：源码编译需要 cu12.8 工具链，实例上只有 12.1
pip install "flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
# 坑：轮子文件名必须符合 PEP 427，随手改名成 fa.whl 之类 pip 会报 invalid wheel filename
pip freeze > freeze_20261024_v2.txt
python experiment/verl_integration/install_eb_patch.py   # 幂等；生成 patches/*.patch 存档
```

## 3. 模型与数据

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir models/qwen2.5-0.5b   # 954MB
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir models/qwen2.5-1.5b   # ~3.1GB 单文件，中断后可从 .incomplete 续传

python experiment/data/make_countdown.py --train-size 8000 --test-size 512 --seed 42 --out-dir data/countdown
python experiment/data/convert_to_verl.py --src data/countdown --out data/verl
# 实测可用的 verl parquet 字段：data_source / prompt(chat messages) / ability /
#   reward_model{style:"rule", ground_truth:JSON} / extra_info
```

## 4. 冒烟清单（W1 执行结果）

1. [x] vLLM 起服务跑通 0.5B —— 经 verl async server，`gpu_memory_utilization=0.3` 时占 6.2GB
2. [x] verl GRPO 配方 + Countdown 奖励跑通 8 步（`runs/smoke_baseline_seed0`，8 行日志）
3. [x] EB 组闭环跑通：控制器逐步改写 `entropy_coeff`/`clip_ratio_high` 并落 JSONL
4. [x] 每步墙钟实测：**0.5B、批 32、G=8、≤512 token → 42–57 s/步**（`timing_s/step`）
       → 原计划「单卡一天完成 1.5B 的 6 次主实验」乐观约 3–4 倍，规模与种子数须按实测重排
5. [ ] vanilla GRPO 预跑 100 步，确认熵坍缩是否显现（`runs/prerun_baseline_05b_s0`，进行中）
6. [x] 命令、报错、显存/耗时留痕在实例 `/root/ebgrpo/{env_create,pip_vllm_0.11,smoke_*}.log`

两点与文档旧假设不同，写作时要按实测改：

- **初始策略熵 ~1.8–2.0 nats**（Qwen2.5-0.5B base + chat 模板 + Countdown 提示），
  所以方法章的熵带不能沿用 `[0.20, 0.40]` 这类占位值，要按初始熵的比例设定（预跑后定）。
- **0.5B 在 4 个数 Countdown 上奖励近似为零**（8 步内 `critic/score/mean` 0.0035–0.005，
  只有格式分），不存在"奖励均值上升"这条验证；实验定位 accordingly 是熵机制演示，
  不是性能学习曲线（作者 2026-10-24 确认保持 4 个数、如实定位为机制演示）。
