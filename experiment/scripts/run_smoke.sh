#!/usr/bin/env bash
# 冒烟测试：走 train.sh 的真实参数路径，只把规模压到 8 步 / 0.5B。
# 用法：bash run_smoke.sh                    # baseline 组
#       GROUP=eb bash run_smoke.sh           # 验证控制器闭环
set -euo pipefail

export ROOT=${ROOT:-/root/ebgrpo}
export GROUP=${GROUP:-baseline}
export SEED=${SEED:-0}
export MODEL=${MODEL:-models/qwen2.5-0.5b}
export BATCH=${BATCH:-32}
export MINI=${MINI:-32}
export MICROB=${MICROB:-8}
export STEPS=${STEPS:-8}
export SAVE_FREQ=${SAVE_FREQ:--1}
export VLLM_MEM=${VLLM_MEM:-0.3}
export OUT=${OUT:-$ROOT/runs/smoke_${GROUP}_seed${SEED}}
# 占位取值，仅为让 eb/fixed 组也能跑通冒烟；正式取值由 W1 预跑校准
export EB_H_LO=${EB_H_LO:-0.2}
export EB_H_HI=${EB_H_HI:-0.4}
export FIXED_COEFF=${FIXED_COEFF:-0.003}

source /opt/conda/etc/profile.d/conda.sh
conda activate ebgrpo
exec bash "$ROOT/experiment/scripts/train.sh"
