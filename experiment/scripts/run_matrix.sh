#!/usr/bin/env bash
# 主对比跑批矩阵：0.5B × 3 组 × 2 seeds，顺序执行（单卡）。
# 取值来自 W1 预跑实测（runs/prerun_baseline_05b_s0）：
#   初始熵 H0≈1.80 nats；reward 起来之前熵先上飘到 2.36，
#   reward 从 0.005 涨到 0.049 后开始坍缩（41 步 1.33 → 46 步 1.05）。
#   故熵带取 H0 的比例 [0.6, 1.1]，两个方向都在带外，控制器两头都要动。
# 墙钟：~60 s/步 → 单次 150 步约 2.5 h，6 次约 15 h（一夜）。
set -uo pipefail

cd /root/ebgrpo
source /opt/conda/etc/profile.d/conda.sh
conda activate ebgrpo

export MODEL=${MODEL:-models/qwen2.5-0.5b}
export BATCH=${BATCH:-32} MINI=${MINI:-32} MICROB=${MICROB:-8}
export STEPS=${STEPS:-150}
export VLLM_MEM=${VLLM_MEM:-0.3}
export EB_RELATIVE=True EB_H_REF=0.0        # 相对带，参考熵由首个观测自标定
export EB_H_LO=${EB_H_LO:-0.6} EB_H_HI=${EB_H_HI:-1.1}
export EB_COEFF_MAX=${EB_COEFF_MAX:-0.01}
export FIXED_COEFF=${FIXED_COEFF:-0.001}    # 开环基线：文献常用的固定小常数

for seed in ${SEEDS:-0 1}; do
  for group in baseline fixed eb; do
    echo "=== $(date +%F_%H:%M:%S) START GROUP=$group SEED=$seed ==="
    GROUP=$group SEED=$seed bash experiment/scripts/train.sh \
      || echo "!!! FAILED group=$group seed=$seed（继续下一个，不阻断矩阵）"
  done
done
echo "=== $(date +%F_%H:%M:%S) MATRIX_DONE ==="
