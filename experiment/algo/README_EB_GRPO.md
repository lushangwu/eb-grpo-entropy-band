# EB-GRPO 在 verl 上的集成说明

> 状态：**挂点已在 verl v0.8.0 上核对并跑通（W1，2026-10-24）**。verl 内部接口
> 随版本变动，本文记录的是该版本实测结论；换 verl 版本时须重新核对
> `install_eb_patch.py` 的锚点，不得凭记忆写死函数签名。

## 1. 算法回顾（对应论文第 4 章）

每个 optimizer step：

```
输入：本步 rollout 的 token 级平均策略熵 H_t（训练指标中已有，零额外前向）
0. 相对带模式下若尚无参考熵：H_ref = H_t（首个观测），带宽折算为
   [r_lo·H_ref, r_hi·H_ref]；绝对带模式跳过本步
1. e_t = controller.error_of(H_t)            # 死区误差
2. I_t = clamp(leak*I_{t-1} + e_t, ±i_clamp) # PI 积分，抗饱和
3. u_t = clip(k_p*e_t + k_i*I_t, -1, +1)
4. entropy_coeff = u_t * coeff_max          # u>0 熵奖励 / u<0 熵惩罚（双向）
5. clip_higher   = 0.2 + max(0, u_t) * 0.08  # 只单向放宽（DAPO 语义）
6. 用 (entropy_coeff, clip_higher) 构造本步策略损失，其余不变
```

对照组只需第 4、5 步换成常数：baseline(0, 0.2)、fixed(c, 0.2)。

**熵带的两种给法**（`eb_band_relative` 开关）：绝对带直接把 nats 写进
`eb_h_lo/eb_h_hi`；相对带把这两个值解释为**初始熵的比例**，参考熵由
`eb_h_ref` 显式给定或取训练中的首个观测熵。W1 实测 0.5B/1.5B base 模型在
Countdown 提示上的初始熵量级不同（0.5B ~1.8–2.0 nats），相对带让两个规模共用
同一组比例，免逐模型标定——这是论文 §4 "少先验调参"主张的具体实现。

对照组只需第 4、5 步换成常数：baseline(0, 0.2)、fixed(c, 0.2)。

## 2. verl 挂点（待核对）

已知线索（调研 agent 从 verl 文档/源码读得）与 **W1 实测结论**：

- GRPO 的熵奖励系数与 DAPO 式非对称裁剪是原生配置：
  `actor_rollout_ref.actor.entropy_coeff`、`clip_ratio_low` / `clip_ratio_high`
  （DAPO recipe 默认 0.2 / 0.28）。v0.8.0 里 `ppo_loss` 与 `core_algos` 每次调用
  都从同一个 `ActorConfig` 对象读这两个值（`verl/workers/utils/losses.py:127`、
  `verl/trainer/ppo/core_algos.py:1315`），所以**改这个对象即可逐步生效**，
  不必动损失内核。
- `ActorConfig` 继承 `BaseConfig`，`__init__` 之后字段被冻结；必须把
  `entropy_coeff` / `clip_ratio_high` 登记进 `ActorConfig._mutable_fields`
  （verl 自己的可变通道），否则赋值抛 `FrozenInstanceError`。

**实际采用：方案 1（配置回调）**，落地为 `install_eb_patch.py` 的 4 处改动——
① `ActorConfig` 增 `eb_*` 字段并放开两个执行器；② 两个 yaml 同步默认值
（hydra struct 校验要求键存在）；③ `engine_workers.init_model` 保存
`ppo_loss` partial 所用的 config 引用；④ `update_actor` 前后成对调用
`eb_step_begin/eb_step_end`。方案 2/3 未采用（前者要改损失内核，后者换框架）。

## 3. 训练日志 schema（论文绘图的数据契约，W1 实测）

`eb_bridge` 每个 optimizer step 往 `$OUT/eb_log.jsonl` 追加一行（三组都写，
字段同构；baseline/fixed 的 `u/error/integral/band_*` 缺省）：

```json
{"step": 1, "entropy": 1.983, "u": -1.0, "entropy_coeff": 0.0, "clip_higher": 0.2,
 "kl_loss": 0.0011, "ppo_kl": 0.0, "clipfrac": 0.0, "clipfrac_lower": 0.0,
 "pg_loss": 0.0128, "grad_norm": 0.74,
 "error": -0.2, "integral": -0.2, "band_lo": 1.0, "band_hi": 1.8, "h_ref": 2.0}
```

- `entropy` 取 verl 的 `actor/entropy_loss`（token 平均熵，**正值**：
  `losses.py` 里是 `policy_loss -= entropy_coeff * entropy_loss`）。
- 熵是 `train_mini_batch` 返回的 non-tensor `metrics` 里逐微批的 `Metric`，
  用 verl 自带 `reduce_metrics` 聚合成标量（直接按 `output.keys()` 取是取不到的）。
- 训练器侧指标（`critic/score/mean`、`response_length/mean`、`timing_s/step` 等）
  不在 actor 返回值里，由 `tools/collect_metrics.py` 从 `console.log` 的
  `step:N - k:v - …` 行解析，并与本文件按步号合并成宽表 CSV——
  那是论文第 5 章图表的唯一数据源。
- `group`/`seed` 由 run 目录名（`<group>_seed<n>`）在合并时补齐。
- 反馈有一步延迟：第 t 步的执行器取值由第 t−1 步的观测熵决定（方法章按采样控制论述）。

## 4. 评测

留出集 Countdown 测试 pass@1：`verl_integration/run_eval.py` 用 vLLM 离线批量生成
+ `data/make_countdown.py` 的 `reward()` 判分，不需要 verl 参与。
注意：W1 实测 0.5B base 模型 4 个数 Countdown 的正确率≈0（只有格式分），
`--temperature` 与采样数按此定位取（默认 0.6 / avg@4 仍可，但要清楚
"通过率不劣"才是本实验的比较口径，不是"通过率高"）。
