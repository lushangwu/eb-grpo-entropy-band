"""verl 自定义奖励函数：Countdown 任务。

对齐 verl v0.8.0 reward_manager/naive.py 的调用签名（源码已核对）：
    compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)
返回 float 或含 "score" 键的 dict（dict 会被 verl 记入 reward_extra_info，
便于训练日志里观察奖励构成）。

ground_truth 来自 convert_to_verl.py 生成的 parquet：JSON 字符串
{"nums": [...], "target": int}。真实判分逻辑复用 data/make_countdown.py，
两处永不重复实现，保证训练与评测判分一致。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

from make_countdown import reward as _countdown_reward  # noqa: E402


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source != "countdown":
        return 0.0
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    score, info = _countdown_reward(solution_str, list(gt["nums"]), int(gt["target"]))
    return {"score": score, "reason": info.get("reason", ""), "expr": info.get("expr", "")}
