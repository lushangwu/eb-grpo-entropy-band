"""熵轨迹绘图工具（论文第 5 章核心图）。

输入：训练日志 JSONL（schema 见 algo/README_EB_GRPO.md §3）
输出：灰阶学术图（300 dpi，14.2 cm 宽，供 Word 按宽插入）

用法：
    python plot_entropy.py \
        --files baseline=runs/baseline_seed0/log.jsonl \
                fixed=runs/fixed_seed0/log.jsonl \
                eb=runs/eb_seed0/log.jsonl \
        --h-lo 0.20 --h-hi 0.40 --smooth 5 \
        --out fig_entropy.png
    python plot_entropy.py --selftest   # 合成数据自检（无需真实日志）
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 灰阶 + 中文字体（SimHei 无粗体字重，层级靠线型与深浅表达，不用 bold）
plt.rcParams["font.sans-serif"] = ["SimHei", "Noto Sans CJK SC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

STYLE = {
    "baseline": {"color": "0.55", "ls": "--", "label": "vanilla GRPO"},
    "fixed": {"color": "0.35", "ls": ":", "label": "固定熵系数"},
    "eb": {"color": "0.05", "ls": "-", "label": "EB-GRPO（本文）"},
}
FIG_W_IN = 14.2 / 2.54  # Word 插图宽 14.2 cm


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["step"])
    return rows


def col(rows, key):
    return [r[key] for r in rows if r.get(key) is not None]


def smooth(xs, w):
    if w <= 1 or len(xs) < w:
        return xs
    out, acc = [], 0.0
    from collections import deque

    q = deque()
    for x in xs:
        q.append(x)
        acc += x
        if len(q) > w:
            acc -= q.popleft()
        out.append(acc / len(q))
    return out


def plot(files, h_lo, h_hi, smooth_w, out_path):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FIG_W_IN, 4.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    ax1.axhspan(h_lo, h_hi, color="0.90", zorder=0, label="目标熵带")
    for name, path in files:
        rows = load_jsonl(path)
        steps = col(rows, "step")
        ent = smooth(col(rows, "entropy"), smooth_w)
        st = STYLE.get(name, {"color": "0.5", "ls": "-", "label": name})
        ax1.plot(steps[: len(ent)], ent, color=st["color"], ls=st["ls"], lw=1.4, label=st["label"])
        if name == "eb":
            u = smooth(col(rows, "u"), smooth_w)
            ax2.plot(steps[: len(u)], u, color="0.05", lw=1.2, label="控制量 u")
    ax1.set_ylabel("策略熵 $H_t$")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    ax1.set_ylim(bottom=0)
    ax2.axhline(0, color="0.7", lw=0.8)
    ax2.set_ylabel("$u_t$")
    ax2.set_xlabel("训练步数")
    ax2.legend(frameon=False, fontsize=8)
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"saved: {out_path}")


def selftest(tmp="/tmp/plot_entropy_selftest"):
    os.makedirs(tmp, exist_ok=True)
    rng = random.Random(0)
    specs = {
        "baseline": lambda t: max(0.02, 0.6 - 0.0055 * t) + rng.gauss(0, 0.02),
        "fixed": lambda t: 0.08 + 0.03 * rng.random(),
        "eb": lambda t: 0.30 + 0.08 * ((t // 30) % 2 * 2 - 1) * 0.5 + rng.gauss(0, 0.02),
    }
    for name, fn in specs.items():
        path = os.path.join(tmp, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for t in range(150):
                row = {"group": name, "step": t, "entropy": max(0.001, fn(t)), "u": None}
                if name == "eb":
                    row["u"] = max(0.0, min(1.0, 0.4 + 0.3 * ((t // 30) % 2)))
                f.write(json.dumps(row) + "\n")
    plot(
        [(n, os.path.join(tmp, f"{n}.jsonl")) for n in ("baseline", "fixed", "eb")],
        0.2,
        0.4,
        5,
        os.path.join(tmp, "fig_entropy.png"),
    )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", help="name=path 形式，可多组")
    ap.add_argument("--h-lo", type=float, default=0.20)
    ap.add_argument("--h-hi", type=float, default=0.40)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--out", type=str, default="fig_entropy.png")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    files = []
    for item in args.files:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--files 需要 name=path 形式，收到 {item!r}")
        files.append((name.strip(), path.strip()))
    plot(files, args.h_lo, args.h_hi, args.smooth, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
