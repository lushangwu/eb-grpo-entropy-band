#!/usr/bin/env bash
# 三组对比实验的统一启动脚本（GROUP/SEED 驱动）。
# 键名按服务器上实测锁定的 verl v0.8.0 + vLLM 0.11.0 核对（2026-10-24）。
# 三组之间只允许 case 分支里的参数不同，其余配置完全共享，防止对照组漂移。
set -euo pipefail

GROUP=${GROUP:?需要 GROUP=baseline|fixed|eb}
SEED=${SEED:-0}
MODEL=${MODEL:-models/qwen2.5-1.5b}
ROOT=${ROOT:-/root/ebgrpo}
OUT=${OUT:-$ROOT/runs/${GROUP}_seed${SEED}}

# 训练规模（写作计划 §5.1：批 64、G=8、≤512 token、每组 100–150 步）
BATCH=${BATCH:-64}
MINI=${MINI:-64}
MICROB=${MICROB:-8}
ROLLOUT_N=${ROLLOUT_N:-8}
MAX_PROMPT=${MAX_PROMPT:-256}
MAX_RESP=${MAX_RESP:-512}
STEPS=${STEPS:-120}
LR=${LR:-1e-6}
SAVE_FREQ=${SAVE_FREQ:--1}      # 需要 checkpoint 做中段评测时设 20
VLLM_MEM=${VLLM_MEM:-0.35}      # 与 FSDP 训练共卡，按显存实测调
OFFLOAD=${OFFLOAD:-False}       # 1.5B 在 24GB 上与 vLLM 共卡吃紧时设 True
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}   # flash-attn 2.8.3.post2 已在 4090 验证；无 FA 时用 sdpa
# 显存相关：单卡同时放 vLLM rollout 与 FSDP 训练，微批按 token 数而非条数界定
# （0.5B 实测：微批 8 条×768 token 的反向需要 2.93 GiB 单次分配，24GB 直接 OOM）
MAXTOK=${MAXTOK:-2048}          # 每个 micro batch 每卡的 token 上限（动态分批）
RMPAD=${RMPAD:-True}            # 变长打包（需 flash-attn），省 padding 浪费
CHUNK_ENTROPY=${CHUNK_ENTROPY:-True}   # 分块算熵，避免整块 logits 常驻
# 注：不要设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True —— vLLM 的
# CuMemAllocator（sleep 模式的显存池）与之互斥，会在 worker 启动时断言失败。

# 三组差异参数（唯一允许不同的地方）
EB_ARGS=()
case "$GROUP" in
  baseline)
    ENTROPY_COEFF=0.0
    CLIP_HIGH=0.2
    EB_ENABLED=False
    ;;
  fixed)
    ENTROPY_COEFF=${FIXED_COEFF:?fixed 组需给 FIXED_COEFF（W1 预跑后定）}
    CLIP_HIGH=0.2
    EB_ENABLED=False
    ;;
  eb)
    ENTROPY_COEFF=${EB_COEFF_INIT:-0.0}   # 初值，第一步之后由控制器接管
    CLIP_HIGH=0.2
    EB_ENABLED=True
    EB_ARGS=(
      "actor_rollout_ref.actor.eb_h_lo=${EB_H_LO:?eb 组需给 EB_H_LO（相对带时为比例，绝对带时为 nats）}"
      "actor_rollout_ref.actor.eb_h_hi=${EB_H_HI:?eb 组需给 EB_H_HI（相对带时为比例，绝对带时为 nats）}"
      "actor_rollout_ref.actor.eb_band_relative=${EB_RELATIVE:-True}"
      "actor_rollout_ref.actor.eb_h_ref=${EB_H_REF:-0.0}"   # 0 = 用首个观测熵自标定
      "actor_rollout_ref.actor.eb_coeff_max=${EB_COEFF_MAX:-0.01}"
      "actor_rollout_ref.actor.eb_eps_high_max=${EB_EPS_MAX:-0.28}"
      "actor_rollout_ref.actor.eb_k_p=${EB_KP:-4.0}"
      "actor_rollout_ref.actor.eb_k_i=${EB_KI:-0.5}"
      "actor_rollout_ref.actor.eb_leak=${EB_LEAK:-0.95}"
      "actor_rollout_ref.actor.eb_i_clamp=${EB_ICLAMP:-8.0}"
    )
    ;;
  *) echo "未知 GROUP=$GROUP" >&2; exit 1;;
esac

mkdir -p "$OUT"
echo "== GROUP=$GROUP SEED=$SEED MODEL=$MODEL STEPS=$STEPS OUT=$OUT"
cd "$ROOT"

# 正式入口（run_smoke.sh / run_matrix.sh）都已先 conda activate；直接手调本脚本时
# 兜底到实例上的环境解释器，避免 "python: command not found" 空跑一场。
PYTHON=${PYTHON:-python}
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=/opt/conda/envs/ebgrpo/bin/python

"$PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$ROOT/data/verl/train.parquet \
  data.val_files=$ROOT/data/verl/test.parquet \
  data.seed="$SEED" \
  data.train_batch_size="$BATCH" \
  data.max_prompt_length="$MAX_PROMPT" \
  data.max_response_length="$MAX_RESP" \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding="$RMPAD" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.model.override_config.attn_implementation="$ATTN_IMPL" \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$MINI" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICROB" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAXTOK" \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking="$CHUNK_ENTROPY" \
  actor_rollout_ref.actor.entropy_checkpointing="$CHUNK_ENTROPY" \
  actor_rollout_ref.actor.data_loader_seed="$SEED" \
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high="$CLIP_HIGH" \
  actor_rollout_ref.actor.calculate_entropy=True \
  actor_rollout_ref.actor.fsdp_config.param_offload="$OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OFFLOAD" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.eb_enabled="$EB_ENABLED" \
  actor_rollout_ref.actor.eb_log_file="$OUT/eb_log.jsonl" \
  "${EB_ARGS[@]}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICROB" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$MAXTOK" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="$VLLM_MEM" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICROB" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAXTOK" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  'trainer.logger=["console"]' \
  trainer.project_name=ebgrpo \
  trainer.experiment_name="${GROUP}_seed${SEED}" \
  trainer.default_local_dir="$OUT" \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.total_training_steps="$STEPS" \
  reward.custom_reward_function.path=$ROOT/experiment/verl_integration/countdown_reward.py \
  reward.custom_reward_function.name=compute_score \
  2>&1 | tee "$OUT/console.log"

echo "RUN_EXIT=${PIPESTATUS[0]}" | tee -a "$OUT/console.log"
