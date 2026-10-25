"""Countdown jsonl -> verl 训练格式（parquet）转换脚本。

schema 完全对齐 verl v0.8.0 官方 examples/data_preprocess/gsm8k.py：
    data_source  : "countdown"
    prompt       : [{"role": "user", "content": <题面>}]
    ability      : "reasoning"
    reward_model : {"style": "rule", "ground_truth": <JSON: nums+target>}
    extra_info   : {split, index, solution}

ground_truth 用 JSON 字符串打包 nums 与 target，自定义奖励函数
（verl_integration/countdown_reward.py）解析后调用 make_countdown.reward()。

用法（在服务器 conda 环境内，依赖 pandas/pyarrow，随 verl 安装）：
    python convert_to_verl.py --src ../data/countdown --out ../data/verl
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd


def convert_split(src_jsonl: str, split: str, out_dir: str) -> str:
    rows = [json.loads(line) for line in open(src_jsonl, encoding="utf-8")]
    records = []
    for d in rows:
        ground_truth = json.dumps({"nums": d["nums"], "target": d["target"]}, ensure_ascii=False)
        records.append(
            {
                "data_source": "countdown",
                "prompt": [{"role": "user", "content": d["prompt"]}],
                "ability": "reasoning",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {"split": split, "index": d["idx"], "solution": d["solution"]},
            }
        )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}.parquet")
    pd.DataFrame(records).to_parquet(out_path)
    print(f"{split}: {len(records)} rows -> {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data/countdown", help="含 train.jsonl/test.jsonl 的目录")
    ap.add_argument("--out", default="../data/verl", help="parquet 输出目录")
    args = ap.parse_args()
    for split in ("train", "test"):
        convert_split(os.path.join(args.src, f"{split}.jsonl"), split, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
