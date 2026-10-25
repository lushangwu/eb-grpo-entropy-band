"""Countdown 任务数据生成器与奖励验证器（纯标准库，CPU 可运行）。

论文实验 v2 的训练/评测数据来源：程序化生成 Countdown 类数字谜题，
答案可机器验证（奖励天然 0/1），序列短（<=512 token），适合单卡小规模
GRPO 训练。生成器保证每道题有解（解由随机表达式树构造而来）。

任务规则（与提示词、验证器三方对齐）：
- 给定 k 个数字（默认 k=4）与目标值 target（1..999）；
- 模型须给出恰好用完所有数字的四则运算表达式，结果等于 target；
- 中间过程允许括号；除数不得为 0。

用法：
    python make_countdown.py --train-size 8000 --test-size 512 --seed 42 \
        --out-dir ./countdown
    python make_countdown.py --selftest        # 本地自检（无需 GPU）

训练奖励（verl 侧自定义 reward 函数调用本文件的 reward()）：
    正确（表达式合法、恰好用完数字、结果等于目标）            -> 1.0
    有 <answer> 标签但表达式错误/无法验证                      -> FORMAT_BONUS(0.05)
    无 <answer> 标签                                          -> 0.0
小的格式奖励用于避免训练初期全零奖励组（组相对优势无信号），
常数在 W1 预跑时可校准。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from fractions import Fraction

TOKEN_RE = re.compile(r"\d+|[+\-*/()]")
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

FORMAT_BONUS = 0.05

PROMPT_TEMPLATE = (
    "Solve this Countdown puzzle.\n"
    "Numbers: {nums}\n"
    "Target: {target}\n\n"
    "Create an arithmetic expression that evaluates to exactly {target}, "
    "using each of the numbers {nums} exactly once. You may use +, -, *, / "
    "(division must be exact) and parentheses.\n"
    "First reason step by step inside <think> ... </think>, then output the "
    "final equation inside <answer> ... </answer>, for example: "
    "<answer>(2+8)*5=50</answer>."
)


# ---------------------------------------------------------------- 验证器

class _Parser:
    """递归下降解析器。只接受整数、四则运算与括号；不允许一元负号。

    allowed 为给定数字的多重集合，解析时逐个消耗，超额使用立即报错。
    """

    def __init__(self, tokens: list[str], allowed: Counter):
        self.toks = tokens
        self.i = 0
        self.allowed = allowed
        self.used: list[int] = []

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self) -> Fraction:
        v = self.expr()
        if self.i != len(self.toks):
            raise ValueError("存在未消费的记号")
        return v

    def expr(self) -> Fraction:
        v = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            w = self.term()
            v = v + w if op == "+" else v - w
        return v

    def term(self) -> Fraction:
        v = self.factor()
        while self.peek() in ("*", "/"):
            op = self.take()
            w = self.factor()
            if op == "*":
                v = v * w
            else:
                if w == 0:
                    raise ZeroDivisionError("除数为 0")
                v = v / w
        return v

    def factor(self) -> Fraction:
        t = self.peek()
        if t == "(":
            self.take()
            v = self.expr()
            if self.take() != ")":
                raise ValueError("括号不匹配")
            return v
        if t is not None and t.isdigit():
            self.take()
            n = int(t)
            if self.used.count(n) >= self.allowed.get(n, 0):
                raise ValueError(f"数字 {n} 超额使用")
            self.used.append(n)
            return Fraction(n)
        raise ValueError(f"非法记号 {t!r}")


def verify(expr: str, nums: list[int], target: int) -> tuple[bool, str]:
    """校验表达式：合法、数字恰好用完（多重集合相等）、结果等于 target。"""
    toks = TOKEN_RE.findall(expr)
    if not toks:
        return False, "空表达式"
    if re.sub(r"\s+", "", expr) != "".join(toks):
        return False, "含非法字符（只允许数字与 + - * / ( )）"
    p = _Parser(toks, Counter(nums))
    try:
        v = p.parse()
    except (ValueError, ZeroDivisionError) as e:
        return False, str(e)
    if Counter(p.used) != Counter(nums):
        return False, "未恰好用完所有给定数字"
    if v != Fraction(target):
        return False, f"结果 {v} 不等于目标 {target}"
    return True, "ok"


def extract_answer(text: str) -> str | None:
    """取最后一个 <answer> 标签内容；模型写 'EXPR=TARGET' 时取等号左侧。"""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    ans = matches[-1].strip()
    if not ans:
        return None
    lhs = ans.split("=")[0].strip()
    return lhs or None


def reward(response: str, nums: list[int], target: int) -> tuple[float, dict]:
    """RLVR 奖励函数：正确 1.0；有合法 answer 标签但错误 FORMAT_BONUS；无标签 0。"""
    expr = extract_answer(response)
    if expr is None:
        return 0.0, {"reason": "no_answer_tag"}
    ok, why = verify(expr, nums, target)
    if ok:
        return 1.0, {"reason": "correct", "expr": expr}
    return FORMAT_BONUS, {"reason": f"wrong: {why}", "expr": expr}


# ---------------------------------------------------------------- 生成器

def _sample_numbers(rng: random.Random, k: int) -> list[int]:
    """70% 概率取 1..13 的小数字，否则取 1..99，偏向经典 Countdown 手感。"""
    out = []
    for _ in range(k):
        if rng.random() < 0.7:
            out.append(rng.randint(1, 13))
        else:
            out.append(rng.randint(1, 99))
    return sorted(out)


def _gen_one(rng: random.Random, k: int, target_hi: int = 999):
    """构造一道有解谜题：随机表达式树收敛为单根，返回 (nums, target, solution)。"""
    nums = _sample_numbers(rng, k)
    items = [(Fraction(n), str(n)) for n in nums]
    rng.shuffle(items)
    while len(items) > 1:
        i, j = rng.sample(range(len(items)), 2)
        a, sa = items[i]
        b, sb = items[j]
        ops = ["+", "-", "*", "/"]
        rng.shuffle(ops)
        merged = None
        for op in ops:
            if op == "+":
                v = a + b
            elif op == "-":
                v = a - b
            elif op == "*":
                v = a * b
            else:
                if b == 0:
                    continue
                v = a / b
            # 经典 Countdown 约束：中间值须为非负整数（除法须整除）
            if v.denominator != 1 or v < 0:
                continue
            merged = (v, f"({sa}{op}{sb})")
            break
        if merged is None:
            return None
        for idx in sorted((i, j), reverse=True):
            del items[idx]
        items.append(merged)
    value, solution = items[0]
    t = int(value)
    if not (1 <= t <= target_hi):
        return None
    return nums, t, solution


def generate(n: int, k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    seen: set = set()
    out: list[dict] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > n * 500:
            raise RuntimeError("生成失败：去重后样例不足，请调整参数")
        res = _gen_one(rng, k)
        if res is None:
            continue
        nums, target, solution = res
        key = (tuple(nums), target)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "idx": len(out),
                "nums": nums,
                "target": target,
                "solution": solution,
                "prompt": PROMPT_TEMPLATE.format(nums=nums, target=target),
            }
        )
    return out


# ---------------------------------------------------------------- 自检与主流程

def selftest() -> int:
    rng_seed = 7
    data = generate(200, k=4, seed=rng_seed)
    for d in data:
        ok, why = verify(d["solution"], d["nums"], d["target"])
        assert ok, f"生成器自解未通过验证: {d['solution']} -> {why}"
    # 干扰用例：必须全部拒绝
    d = data[0]
    bad_cases = [
        (d["solution"].replace(str(d["nums"][0]), str(d["nums"][0] + 1) if d["nums"][0] + 1 != d["nums"][1] else str(d["nums"][0] + 2)), "数字替换"),
        (d["solution"] + "+1", "多用数字"),
        (d["solution"][:-1], "截断表达式"),
        ("1+1", "未用完数字"),
        ("".join(reversed(d["solution"])), "非法记号"),
        (d["solution"] + "/0", "除零"),
    ]
    for expr, tag in bad_cases:
        ok, why = verify(expr, d["nums"], d["target"])
        assert not ok, f"干扰用例未被拒绝（{tag}）: {expr}"
    # 奖励函数
    good = f"<think>try</think><answer>{data[0]['solution']}={data[0]['target']}</answer>"
    r, info = reward(good, data[0]["nums"], data[0]["target"])
    assert r == 1.0 and info["reason"] == "correct", f"正确答案奖励异常: {r} {info}"
    r, _ = reward("<think>hmm</think><answer>1+1=2</answer>", data[0]["nums"], data[0]["target"])
    assert r == FORMAT_BONUS, "错误答案应只得格式分"
    r, _ = reward("no tags at all", data[0]["nums"], data[0]["target"])
    assert r == 0.0, "无标签应得 0"
    r, _ = reward("<answer></answer>", data[0]["nums"], data[0]["target"])
    assert r == 0.0, "空标签应得 0"
    print(f"selftest PASS: 200 题全部有解且可验证；干扰用例全部拒绝；奖励函数 4 项通过")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Countdown 数据生成器")
    ap.add_argument("--train-size", type=int, default=8000)
    ap.add_argument("--test-size", type=int, default=512)
    ap.add_argument("--numbers", type=int, default=4, help="每题数字个数 k")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="./countdown")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    os.makedirs(args.out_dir, exist_ok=True)
    targets = {"target_len": []}
    for split, size, seed in (("train", args.train_size, args.seed), ("test", args.test_size, args.seed + 1)):
        data = generate(size, k=args.numbers, seed=seed)
        path = os.path.join(args.out_dir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        lens = [d["target"] for d in data]
        targets["target_len"].append((split, min(lens), max(lens)))
        print(f"{split}: {len(data)} 题 -> {path}")
    for split, lo, hi in targets["target_len"]:
        print(f"  {split} 目标值范围: {lo}..{hi}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
