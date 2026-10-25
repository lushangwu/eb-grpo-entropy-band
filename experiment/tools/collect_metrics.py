"""把一次训练的 verl console 日志与 EB 日志合并成逐步骤宽表 CSV。

verl 的 console logger 每步打一行 `step:N - k:v - k:v ...`（只含数值项，
见 verl/utils/logger/aggregate_logger.py:concat_dict_to_str），里面有
reward/响应长度/KL 等训练器侧指标；EB 桥的 eb_log.jsonl 里有控制器字段
（u、熵带误差、实际控制的 entropy_coeff 与 clip_higher）。两者按步号合并，
得到论文第 5 章所有曲线与表格的唯一数据源。

用法：
    python collect_metrics.py --runs runs/baseline_seed0 runs/fixed_seed0 ... \
        --out results/main_metrics.csv
    python collect_metrics.py --dir runs --out results/main_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

STEP_LINE = re.compile(r"step:(\d+) - ")


def parse_console(path: str) -> dict[int, dict[str, float]]:
    """解析 verl console logger 的每步指标行。

    注意：ray 会给工作进程的输出加 `(TaskRunner pid=N) ` 前缀，所以指标行不一定
    在行首；这里用 search 定位 `step:N - ` 的起点，再按 " - " 切分后面的 k:v 对。
    """
    steps: dict[int, dict[str, float]] = {}
    if not os.path.exists(path):
        return steps
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STEP_LINE.search(line)
            if not m:
                continue
            rec: dict[str, float] = {}
            for item in line[m.end():].split(" - "):
                key, sep, val = item.partition(":")
                if not sep:
                    continue
                try:
                    rec[key.strip()] = float(val)
                except ValueError:
                    continue
            if rec:
                steps[int(m.group(1))] = rec
    return steps


def parse_eb_log(path: str) -> dict[int, dict]:
    recs: dict[int, dict] = {}
    if not os.path.exists(path):
        return recs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in r:
                recs[int(r["step"])] = r
    return recs


def run_name(path: str) -> tuple[str, str]:
    """从目录名推断 (group, seed)；取不到就用目录名。"""
    base = os.path.basename(os.path.normpath(path))
    m = re.match(r"(?:smoke_)?([a-z0-9]+)_seed(\d+)", base)
    if m:
        return m.group(1), m.group(2)
    return base, ""


def collect(run_dir: str) -> list[dict]:
    console = parse_console(os.path.join(run_dir, "console.log"))
    eb = parse_eb_log(os.path.join(run_dir, "eb_log.jsonl"))
    group, seed = run_name(run_dir)
    rows = []
    for step in sorted(set(console) | set(eb)):
        row = {"run": os.path.basename(os.path.normpath(run_dir)), "group": group, "seed": seed, "step": step}
        row.update(console.get(step, {}))
        for k, v in eb.get(step, {}).items():
            if k != "step":
                row[f"eb_{k}" if k in row else k] = v
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="一个或多个 run 目录")
    ap.add_argument("--dir", default="", help="批量收集该目录下的 run 子目录（含 console.log 或 eb_log.jsonl）")
    ap.add_argument("--out", default="results/main_metrics.csv")
    args = ap.parse_args()

    dirs = list(args.runs)
    if args.dir:
        for cand in sorted(glob.glob(os.path.join(args.dir, "*"))):
            if os.path.isdir(cand) and (
                os.path.exists(os.path.join(cand, "console.log")) or os.path.exists(os.path.join(cand, "eb_log.jsonl"))
            ):
                dirs.append(cand)
    if not dirs:
        raise SystemExit("没有可用的 run 目录（用 --runs 指定，或 --dir 指向 runs 根目录）")

    all_rows = []
    for d in dirs:
        rows = collect(d)
        print(f"{d}: {len(rows)} 步")
        all_rows.extend(rows)

    fields: list[str] = []
    for row in all_rows:
        for k in row:
            if k not in fields:
                fields.append(k)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"-> {args.out}（{len(all_rows)} 行，{len(fields)} 列）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
