"""从宽表算论文第 5 章的聚合指标，输出 results/summary.csv 与控制台表格。

评价口径（对三组统一，才可比）：
- 参考熵 H0 = 该 run 第 1 步的 token 平均熵；评价带 = [0.6·H0, 1.1·H0]
  （即 EB 组的控制目标，同样作为 baseline/fixed 的"是否守住探索水平"的标尺）。
- 带内比例：第 6 步起，熵落在评价带内的步数占比（跳过自标定前的首几步）。
- 熵 AUC：前 N 步熵的梯形积分 / N，衡量"整个训练过程保住了多少探索"。
- 末期熵 / 最低熵：末 20 步均值、全程最小值（坍缩深度）。
- score：末 20 步均值与峰值。注意本实验里 score 上限 0.05 全为格式分，
  任务正确率为 0，所以 score 只用于"不劣"检验，不作性能主张。
- grad_norm 末 20 步均值：探索死亡的直接读数。
- 控制量：|u|>0.05 的步数占比、u 的符号分布（只报 eb 组）。

用法：python summarize.py --csv results/main_metrics.csv --out results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict

GROUPS = ("baseline", "fixed", "eb")


def load(path: str):
    runs = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("group") not in GROUPS:
                continue
            runs[(row["group"], row["seed"])].append(row)
    for key in runs:
        runs[key].sort(key=lambda r: int(r["step"]))
    return runs


def num(row, key):
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def summarize(rows):
    ent = [num(r, "entropy") for r in rows]
    ent = [e for e in ent if e is not None]
    n = len(ent)
    if n == 0:
        return None
    h0 = ent[0]
    lo, hi = 0.6 * h0, 1.1 * h0
    tail = ent[-20:]
    inside = sum(1 for e in ent[5:] if lo <= e <= hi)
    auc = sum((ent[i] + ent[i + 1]) / 2 for i in range(n - 1)) / max(1, n - 1)

    def mean_tail(key):
        vals = [num(r, key) for r in rows[-20:]]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    out = {
        "steps": n,
        "H0": h0,
        "band_lo": lo,
        "band_hi": hi,
        "H_tail_mean": sum(tail) / len(tail),
        "H_min": min(ent),
        "H_auc": auc,
        "band_frac": inside / max(1, n - 5),
        "score_tail_mean": mean_tail("critic/score/mean"),
        "score_max": max((v for v in (num(r, "critic/score/mean") for r in rows) if v is not None), default=None),
        "kl_tail_mean": mean_tail("actor/kl_loss"),
        "len_tail_mean": mean_tail("response_length/mean"),
        "clip_ratio_tail": mean_tail("response_length/clip_ratio"),
        "grad_tail_mean": mean_tail("grad_norm"),
    }
    u = [num(r, "u") for r in rows[5:]]
    u = [v for v in u if v is not None]
    if u:
        out["active_frac"] = sum(1 for v in u if abs(v) > 0.05) / len(u)
        out["u_pos_frac"] = sum(1 for v in u if v > 0.05) / len(u)
        out["u_neg_frac"] = sum(1 for v in u if v < -0.05) / len(u)
        out["u_saturated_frac"] = sum(1 for v in u if abs(v) >= 0.995) / len(u)
    return out


def fmt(v, spec="%.3f"):
    return "-" if v is None else spec % v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/main_metrics.csv")
    ap.add_argument("--out", default="results/summary.csv")
    args = ap.parse_args()

    runs = load(args.csv)
    rows_out = []
    for (group, seed), rows in sorted(runs.items()):
        s = summarize(rows)
        if not s:
            continue
        rec = {"group": group, "seed": seed}
        rec.update(s)
        rows_out.append(rec)

    cols = ["group", "seed", "steps", "H0", "band_lo", "band_hi", "H_tail_mean", "H_min", "H_auc",
            "band_frac", "score_tail_mean", "score_max", "kl_tail_mean", "len_tail_mean",
            "clip_ratio_tail", "grad_tail_mean", "active_frac", "u_pos_frac", "u_neg_frac", "u_saturated_frac"]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    print(f"-> {args.out}（{len(rows_out)} 个 run）\n")
    pct = lambda v: "-" if v is None else f"{100*v:.0f}%"
    head = (f"{'组':<10}{'种子':<5}{'末期熵':>9}{'最低熵':>9}{'熵AUC':>9}{'带内':>6}"
            f"{'score':>8}{'KL':>8}{'长度':>7}{'gradN':>8}{'动作':>6}{'饱和':>6}")
    print(head)
    for group in GROUPS:
        recs = [r for r in rows_out if r["group"] == group]
        for r in recs:
            print(f"{group:<10}{r['seed']:<5}{fmt(r['H_tail_mean']):>9}{fmt(r['H_min']):>9}"
                  f"{fmt(r['H_auc']):>9}{pct(r['band_frac']):>6}{fmt(r['score_tail_mean'], '%.4f'):>8}"
                  f"{fmt(r['kl_tail_mean'], '%.4f'):>8}{fmt(r['len_tail_mean'], '%.0f'):>7}"
                  f"{fmt(r['grad_tail_mean']):>8}{pct(r.get('active_frac')):>6}"
                  f"{pct(r.get('u_saturated_frac')):>6}")
        if len(recs) > 1:
            keys = ("H_tail_mean", "H_min", "H_auc", "band_frac", "score_tail_mean",
                    "kl_tail_mean", "len_tail_mean", "grad_tail_mean")
            avg = {}
            for k in keys:
                vals = [r[k] for r in recs if r.get(k) is not None]
                avg[k] = sum(vals) / len(vals) if vals else None
            print(f"{group} 均值 {'':<4}{fmt(avg['H_tail_mean']):>9}{fmt(avg['H_min']):>9}"
                  f"{fmt(avg['H_auc']):>9}{pct(avg['band_frac']):>6}{fmt(avg['score_tail_mean'], '%.4f'):>8}"
                  f"{fmt(avg['kl_tail_mean'], '%.4f'):>8}{fmt(avg['len_tail_mean'], '%.0f'):>7}"
                  f"{fmt(avg['grad_tail_mean']):>8}{'':>6}{'':>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
