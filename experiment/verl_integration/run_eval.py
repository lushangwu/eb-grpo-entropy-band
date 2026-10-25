"""离线评测：vLLM 批量生成 + Countdown 奖励判分。

用于：① 训练前基线摸底；② 训练中每隔 N 步评 checkpoint（pass@1 曲线）；
③ 最终论文表格数据。判分与训练奖励同源（make_countdown.reward）。

用法（服务器 conda 环境内）：
    python run_eval.py --model models/qwen2.5-0.5b --data data/countdown/test.jsonl \
        --n-samples 4 --temperature 0.6 --limit 128 --out evals/base_0.5b.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

from make_countdown import reward as _countdown_reward  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/countdown/test.jsonl")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 题；0=全部")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": d["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        for d in rows
    ]

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem, max_model_len=1024)
    sp = SamplingParams(
        n=args.n_samples, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )
    outs = llm.generate(prompts, sp)

    per_q = []
    n_correct_total = 0
    n_fmt_total = 0
    n_gen_total = 0
    for d, out in zip(rows, outs):
        scores = []
        for o in out.outputs:
            r, info = _countdown_reward(o.text, list(d["nums"]), int(d["target"]))
            scores.append(r)
            n_gen_total += 1
            if r == 1.0:
                n_correct_total += 1
            if r > 0:
                n_fmt_total += 1
        per_q.append({"idx": d["idx"], "target": d["target"], "scores": scores})

    result = {
        "model": args.model,
        "data": args.data,
        "n_questions": len(rows),
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "pass_at_1_mean_per_question": sum(sum(q["scores"]) / len(q["scores"]) for q in per_q) / len(per_q),
        "overall_correct_rate": n_correct_total / max(1, n_gen_total),
        "format_valid_rate": n_fmt_total / max(1, n_gen_total),
        "per_question": per_q,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(
        f"pass@1(按题均值)={result['pass_at_1_mean_per_question']:.4f} "
        f"总体正确率={result['overall_correct_rate']:.4f} 格式有效率={result['format_valid_rate']:.4f} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
