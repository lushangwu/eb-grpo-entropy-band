"""论文第 5 章数据图（图 2–图 4），灰阶学术风格，单栏 14.2 cm。

数据源：results/main_metrics.csv（由 collect_metrics.py 合并逐步骤全指标）。
风格约定：只用中性灰阶 + 线型/标记区分方法族（不用彩色）；中文用 SimHei 且必须
排在字体栈首位（Arial 在前时汉字不回退），SimHei 无粗体字重，故仅面板号用
Arial 显式加粗。
导出：PDF（矢量、可编辑文字）+ 600 dpi PNG（Word 按 14.2 cm 宽插入）。

用法（在课程论文作业目录下）：
    python experiment/tools/make_paper_figures.py
面板对齐门需要技能脚本目录在路径里：
    set PYTHONPATH=<figure 技能>/scripts 后再运行，否则记为 NOT AUDITABLE。
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── 字体与全局样式（MANDATORY：SVG/PDF 文字可编辑）────────────────────────
# SimHei 必须排在首位：把 Arial 放前面时 matplotlib 不会为汉字回退（实测出方框）。
# SimHei 无粗体字重，故面板号单独用 Arial 显式加粗，正文一律不加粗。
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans", "Arial"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.6,
    "axes.labelsize": 7.6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
    "axes.unicode_minus": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
})

# 灰阶方法族：baseline 浅、fixed 中、EB 深（本文方法最显著）
STYLE = {
    "baseline": {"color": "0.62", "ls": "--", "lw": 1.3, "label": "vanilla GRPO"},
    "fixed": {"color": "0.42", "ls": ":", "lw": 1.3, "label": "GRPO + 固定熵系数"},
    "eb": {"color": "0.05", "ls": "-", "lw": 1.6, "label": "EB-GRPO（本文）"},
}
GROUPS = ("baseline", "fixed", "eb")
BAND_LO_R, BAND_HI_R = 0.6, 1.1          # 评价带 = 初始熵的比例
WIDTH_CM = 14.2
IN = WIDTH_CM / 2.54
OUT_DIR = "figures"


def blend_xy(ax):
    """横向按 axes 分数、纵向按数据坐标定位文本（避免两套坐标混用时跑偏）。"""
    from matplotlib.transforms import blended_transform_factory
    return blended_transform_factory(ax.transAxes, ax.transData)


def add_panel_label(ax, label, x=0, y=1, x_offset_pt=-3.5, y_offset_pt=3, fontsize=8):
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import ScaledTranslation
    off = ScaledTranslation(x_offset_pt / 72, y_offset_pt / 72, ax.figure.dpi_scale_trans)
    ax.text(x, y, label, transform=ax.transAxes + off, fontsize=fontsize,
            fontproperties=FontProperties(family="Arial", weight="bold"),
            ha="left", va="bottom", clip_on=False)


def load_runs(path: str):
    runs = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("group") not in GROUPS:
                continue
            runs[(row["group"], row["seed"])].append(row)
    for key in runs:
        runs[key].sort(key=lambda r: int(r["step"]))
    return runs


def series(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out)


def steps(rows):
    return np.array([int(r["step"]) for r in rows], dtype=float)


def smooth(x, w=5):
    if w <= 1 or len(x) < w:
        return x
    k = np.ones(w) / w
    pad = np.concatenate([np.full(w // 2, x[0]), x, np.full(w - 1 - w // 2, x[-1])])
    return np.convolve(pad, k, mode="valid")[: len(x)]


def align(fig, name, **kw):
    """面板对齐门；技能脚本不在路径里时记 NOT AUDITABLE（不静默跳过）。"""
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        print(f"[QA] {name}: NOT AUDITABLE（缺 audit_panel_alignment，设 PYTHONPATH 后重跑）")
        return
    require_matplotlib_panel_alignment(
        fig, json_out=os.path.join(OUT_DIR, f"{name}.alignment.json"),
        overlay_svg=os.path.join(OUT_DIR, f"{name}.alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, **kw)
    print(f"[QA] {name}: 对齐门通过")


def export(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=600)
    plt.close(fig)
    print(f"saved: {name} (pdf/png)")


# ── 图 2：主结果（hero = 熵轨迹）──────────────────────────────────────────
def fig2(runs):
    fig = plt.figure(figsize=(IN, 5.05))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1], hspace=0.42, wspace=0.42,
                          left=0.085, right=0.985, top=0.955, bottom=0.085)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    # a：熵轨迹，种子各一条（细）+ 组均值（粗）+ 评价带
    h0_all = []
    for g in GROUPS:
        st = STYLE[g]
        per_seed = {}
        for (grp, seed), rows in runs.items():
            if grp != g:
                continue
            e = series(rows, "entropy")
            h0_all.append(e[0])
            per_seed[seed] = e
            ax_a.plot(steps(rows), e, color=st["color"], ls=st["ls"], lw=0.7,
                      alpha=0.5, zorder=2)
        mean = np.mean(np.vstack([v for v in per_seed.values()]), axis=0)
        x_ref = steps([rows for (grp, _), rows in runs.items() if grp == g][0])
        ax_a.plot(x_ref, smooth(mean, 5), color=st["color"], ls=st["ls"], lw=st["lw"],
                  label=st["label"], zorder=4)
    # 评价带按六个 run 的平均初始熵折算（各组 H0 同量级，图注说明）
    h0 = float(np.mean(h0_all))
    ax_a.axhspan(BAND_LO_R * h0, BAND_HI_R * h0, color="0.86", zorder=0)
    ax_a.set_xlabel("训练步数")
    ax_a.set_ylabel("token 平均策略熵 $H_t$（nats）")
    ax_a.set_ylim(0, 2.9)
    from matplotlib.patches import Patch
    handles, leg_labels = ax_a.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.86", label=f"评价带 [{BAND_LO_R:.1f}, {BAND_HI_R:.1f}]×H$_0$"))
    ax_a.legend(handles=handles, loc="lower left", fontsize=7.2, ncol=4,
                handlelength=1.8, columnspacing=1.1, borderaxespad=0.1,
                handletextpad=0.5)
    add_panel_label(ax_a, "a")

    # b：EB 的控制量 u 与执行器 entropy_coeff
    for (grp, seed), rows in sorted(runs.items()):
        if grp != "eb":
            continue
        x = steps(rows)
        ax_b.plot(x, series(rows, "u"), color="0.05" if seed == "0" else "0.45",
                  ls="-" if seed == "0" else "-.", lw=1.0, label=f"seed {seed}")
    ax_b.axhspan(-0.05, 0.05, color="0.90", zorder=0)
    ax_b.axhline(1, color="0.5", lw=0.6, ls=":")
    ax_b.axhline(-1, color="0.5", lw=0.6, ls=":")
    ax_b.set_ylim(-1.45, 1.45)
    ax_b.set_xlabel("训练步数")
    ax_b.set_ylabel("控制量 $u_t$")
    # ±1 处的点线即执行器饱和边界，由 y 轴刻度读数，不再加文字标注（会与曲线相撞）
    ax_b.legend(loc="upper left", bbox_to_anchor=(0.02, 0.99), ncol=2, fontsize=7.2,
                handlelength=1.2, columnspacing=0.9, borderaxespad=0)
    add_panel_label(ax_b, "b")

    # c：梯度范数（探索死亡的直接读数）
    for g in GROUPS:
        st = STYLE[g]
        for (grp, seed), rows in runs.items():
            if grp != g:
                continue
            ax_c.plot(steps(rows), smooth(series(rows, "grad_norm"), 9),
                      color=st["color"], ls=st["ls"],
                      lw=st["lw"] if seed == "0" else 0.8,
                      alpha=1.0 if seed == "0" else 0.55,
                      label=st["label"] if seed == "0" else None)
    ax_c.set_xlabel("训练步数")
    ax_c.set_ylabel("$\\Vert\\nabla J\\Vert$（末9步滑动均值）")
    ax_c.legend(loc="upper right", fontsize=5.8, handlelength=1.6)
    add_panel_label(ax_c, "c")

    # d：奖励（限定主张：三组同卡格式分上限）
    for g in GROUPS:
        st = STYLE[g]
        for (grp, seed), rows in runs.items():
            if grp != g:
                continue
            ax_d.plot(steps(rows), smooth(series(rows, "critic/score/mean"), 9),
                      color=st["color"], ls=st["ls"],
                      lw=st["lw"] if seed == "0" else 0.8,
                      alpha=1.0 if seed == "0" else 0.55,
                      label=st["label"] if seed == "0" else None)
    ax_d.axhline(0.05, color="0.3", lw=0.7, ls=(0, (4, 2)))
    ax_d.text(0.97, 0.0515, "格式分上限 0.05", transform=blend_xy(ax_d), ha="right",
              va="bottom", fontsize=5.8, color="0.25")
    ax_d.set_ylim(0, 0.062)
    ax_d.set_xlabel("训练步数")
    ax_d.set_ylabel("组内平均奖励")
    add_panel_label(ax_d, "d")

    align(fig, "fig2_main", axes=[ax_a], panel_ids=["a"], exclude_axes=[ax_b, ax_c, ax_d],
          exemptions=[{"panels": ["a"], "checks": ["column", "row", "panel-width"],
                       "reason": "hero panel spans all three grid columns by design"}])
    export(fig, "fig2_main")


# ── 图 3：配对差与种子离散 ────────────────────────────────────────────────
def aggregates(rows):
    e = series(rows, "entropy")
    e = e[~np.isnan(e)]
    h0 = e[0]
    lo, hi = BAND_LO_R * h0, BAND_HI_R * h0
    tail = lambda k: float(np.nanmean(series(rows, k)[-20:]))
    inside = float(np.mean((e[5:] >= lo) & (e[5:] <= hi)))
    auc = float(np.mean((e[:-1] + e[1:]) / 2))
    return {"band_frac": inside, "auc": auc, "h_tail": tail("entropy"),
            "grad_tail": tail("grad_norm")}


def fig3(runs):
    agg = {key: aggregates(rows) for key, rows in runs.items()}
    metrics = [("band_frac", "带内比例"), ("auc", "熵 AUC"),
               ("h_tail", "末期熵"), ("grad_tail", "末期梯度范数")]
    fig = plt.figure(figsize=(IN, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.72,
                          left=0.135, right=0.975, top=0.90, bottom=0.24)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # a：按种子配对的倍率（treated ÷ baseline，对数轴）。
    # 用倍率而非百分比变化：带内比例的 baseline 只有 0.10，百分比会放大到 +500%
    # 而熵 AUC 只有 +45%，同一根轴上小效应会被压平；对数倍率能同框容纳两个量级。
    labels, eb_ratio, fx_ratio = [], [], []
    for key, zh in metrics:
        er, fr = [], []
        for seed in ("0", "1"):
            b = agg[("baseline", seed)][key]
            er.append(agg[("eb", seed)][key] / b)
            fr.append(agg[("fixed", seed)][key] / b)
        labels.append(zh)
        eb_ratio.append(er)
        fx_ratio.append(fr)
    y = np.arange(len(labels))[::-1]
    h = 0.16
    for i, yi in enumerate(y):
        for vals, off, st in ((eb_ratio[i], h, STYLE["eb"]), (fx_ratio[i], -h, STYLE["fixed"])):
            ax_a.plot([min(vals), max(vals)], [yi + off] * 2, color=st["color"], lw=2.2,
                      solid_capstyle="butt")
            ax_a.plot(vals, [yi + off] * 2, marker="o", ms=3.2, color=st["color"],
                      mec="white", mew=0.5, ls="none")
        ax_a.plot([1, 1], [yi - 0.30, yi + 0.30], color="0.75", lw=0.7)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=6.5)
    ax_a.set_xscale("log")
    ax_a.set_xlim(0.85, 12)
    ax_a.set_xticks([1, 2, 4, 8])
    ax_a.set_xticklabels(["1", "2", "4", "8"])
    ax_a.minorticks_off()
    ax_a.axvline(1, color="0.2", lw=0.8)
    ax_a.set_xlabel("相对 vanilla GRPO 的倍率（对数轴，逐种子配对）")
    # 直接标注两条线族的归属，省掉图例（顶行右侧留白足够）
    ax_a.text(max(eb_ratio[0]) * 1.12, y[0] + h, "EB-GRPO", fontsize=5.8,
              color=STYLE["eb"]["color"], va="center")
    ax_a.text(max(fx_ratio[0]) * 1.12, y[0] - h, "固定熵系数", fontsize=5.8,
              color=STYLE["fixed"]["color"], va="center")
    add_panel_label(ax_a, "a")

    # b：末期熵 × 带内比例，六个 run 的散点（看组间分离是否盖过种子离散）
    for g in GROUPS:
        st = STYLE[g]
        for seed in ("0", "1"):
            a = agg[(g, seed)]
            ax_b.scatter(a["band_frac"] * 100, a["h_tail"], s=26, facecolor=st["color"],
                         edgecolor="white", linewidth=0.6, marker="o" if seed == "0" else "s",
                         zorder=3)
    ax_b.annotate("EB-GRPO", xy=(58, 1.40), xytext=(30, 2.10), fontsize=6,
                  ha="center", arrowprops=dict(arrowstyle="->", lw=0.6, color="0.3"))
    ax_b.annotate("vanilla GRPO", xy=(10, 0.43), xytext=(26, 0.30), fontsize=6,
                  ha="center", arrowprops=dict(arrowstyle="->", lw=0.6, color="0.3"))
    ax_b.set_xlabel("带内比例（%）")
    ax_b.set_ylabel("末期熵（nats）")
    ax_b.set_xlim(0, 70)
    ax_b.set_ylim(0.2, 2.3)
    ax_b.text(0.02, 0.98, "○ seed0  □ seed1", transform=ax_b.transAxes, ha="left",
              va="top", fontsize=5.8, color="0.25")
    add_panel_label(ax_b, "b")

    align(fig, "fig3_effect", axes=[ax_a, ax_b], panel_ids=["a", "b"],
          row_groups=[["a", "b"]],
          exemptions=[{"panels": ["a", "b"], "checks": ["panel-width"],
                       "reason": "不等宽双栏：森林图需要更宽的横轴容纳四个指标"}])
    export(fig, "fig3_effect")


# ── 图 4：失效边界（饱和—过冲极限环）─────────────────────────────────────
def fig4(runs):
    rows = runs[("eb", "1")]
    x = steps(rows)
    e = series(rows, "entropy")
    u = series(rows, "u")
    coeff = series(rows, "entropy_coeff")
    lo, hi = rows[-1]["band_lo"], rows[-1]["band_hi"]
    lo, hi = float(lo), float(hi)

    fig = plt.figure(figsize=(IN, 2.75))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.34,
                          left=0.105, right=0.98, top=0.94, bottom=0.165)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])

    ax_a.axhspan(lo, hi, color="0.88", zorder=0)
    ax_a.plot(x, e, color="0.05", lw=1.4, zorder=3)
    # 带外区间着色：下方=探索不足，上方=过度随机
    below = e < lo
    above = e > hi
    ax_a.fill_between(x, 0, e, where=below, color="0.55", alpha=0.35, zorder=1,
                      lw=0, step=None)
    ax_a.fill_between(x, hi, e, where=above, color="0.30", alpha=0.30, zorder=1, lw=0)
    sat = np.abs(u) >= 0.995
    for i in np.where(sat & below)[0]:
        ax_a.plot(x[i], e[i], marker="v", ms=2.6, color="0.05", zorder=4)
    from matplotlib.transforms import blended_transform_factory
    bl = blended_transform_factory(ax_a.transAxes, ax_a.transData)
    ax_a.text(0.02, hi + 0.06, "带上界", transform=bl, ha="left", va="bottom",
              fontsize=6, color="0.3")
    ax_a.text(0.99, lo - 0.06, "带下界", transform=bl, ha="right", va="top",
              fontsize=6, color="0.3")
    ax_a.annotate("过冲至上界外", xy=(90, 2.47), xytext=(96, 2.62), fontsize=6,
                  arrowprops=dict(arrowstyle="->", lw=0.6, color="0.25"))
    ax_a.text(0.015, 0.97, "灰块 = 带外　▼ = 执行器饱和", transform=ax_a.transAxes,
              ha="left", va="top", fontsize=6, color="0.3")
    ax_a.set_ylim(0, 3.0)
    ax_a.set_ylabel("$H_t$（nats）")
    ax_a.set_xticklabels([])
    add_panel_label(ax_a, "a")

    ax_b.plot(x, u, color="0.05", lw=1.2, label="$u_t$")
    ax_b.axhline(1, color="0.4", lw=0.6, ls=":")
    ax_b.axhline(-1, color="0.4", lw=0.6, ls=":")
    ax_b.axhline(0, color="0.75", lw=0.6)
    ax_b.fill_between(x, -1.25, 1.25, where=sat, color="0.62", alpha=0.35, lw=0,
                      step=None, label="执行器饱和")
    ax_b.set_ylim(-1.25, 1.25)
    ax_b.set_xlabel("训练步数")
    ax_b.set_ylabel("$u_t$")
    ax_b.legend(loc="upper left", fontsize=7.2, handlelength=1.6, borderaxespad=0.35)
    add_panel_label(ax_b, "b")

    align(fig, "fig4_boundary", axes=[ax_a, ax_b], panel_ids=["a", "b"],
          column_groups=[["a", "b"]])
    export(fig, "fig4_boundary")


# ── 图 1：机理链 + 闭环控制回路（示意，非数据图）─────────────────────────
def _box(ax, x, y, w, h, text, fc="0.96", ec="0.25", fontsize=6.6, style="-"):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=0.8, edgecolor=ec, facecolor=fc, linestyle=style,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=3)


def _arrow(ax, x1, y1, x2, y2, style="-|>", ls="-", color="0.2", lw=0.9):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, linestyle=ls, color=color, lw=lw,
                                shrinkA=1.5, shrinkB=1.5), zorder=1)


def fig1():
    fig = plt.figure(figsize=(IN, 3.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.42,
                          left=0.012, right=0.988, top=0.955, bottom=0.03)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    for ax in (ax_a, ax_b):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2.3)
        ax.axis("off")

    BW, BH, GX = 2.62, 0.78, 0.62          # 框宽按 8 字标签 + 6.4pt 留足余量
    xs = [0.10, 0.10 + BW + GX, 0.10 + 2 * (BW + GX)]

    # a：可验证奖励下的熵失控因果链（蛇形：上排左→右，下排右→左）
    top = [("组内奖励同质化", "0.94"), ("cov(A, log π)>0", "0.86"), ("策略熵 H 下降", "0.74")]
    bottom = [("奖励平台锁死", "0.94"), ("优势趋零", "0.60"), ("探索死亡", "0.48")]
    for i, (txt, fc) in enumerate(top):
        _box(ax_a, xs[i], 1.42, BW, BH, txt, fc=fc)
    for i, (txt, fc) in enumerate(bottom):
        _box(ax_a, xs[i], 0.18, BW, BH, txt, fc=fc)
    for i in range(2):
        _arrow(ax_a, xs[i] + BW, 1.81, xs[i + 1], 1.81)
    _arrow(ax_a, xs[2] + BW / 2, 1.42, xs[2] + BW / 2, 0.96)
    for i in (2, 1):
        _arrow(ax_a, xs[i], 0.57, xs[i - 1] + BW, 0.57)
    _arrow(ax_a, xs[0] + BW / 2, 0.96, xs[0] + BW / 2, 1.42, ls="--", color="0.4")
    ax_a.text(xs[0] + BW / 2 + 0.10, 1.19, "闭环", fontsize=6.2, color="0.35", va="center")
    add_panel_label(ax_a, "a")

    # b：EB-GRPO 闭环回路（上排前向，下排反馈）
    f_top = [("策略 π$_θ$", "0.98"), ("Rollout：G 个回复", "0.98"), ("可验证奖励\n正确性 + 格式", "0.98")]
    f_bot = [("GRPO 更新 式(2)(3)", "0.90"), ("双执行器 式(8)(9)", "0.86"), ("死区+PI 式(5)(6)\n观测熵 H$_t$", "0.86")]
    for i, (txt, fc) in enumerate(f_top):
        _box(ax_b, xs[i], 1.42, BW, BH, txt, fc=fc)
    for i, (txt, fc) in enumerate(f_bot):
        _box(ax_b, xs[i], 0.18, BW, BH, txt, fc=fc, style="--" if i == 1 else "-")
    for i in range(2):
        _arrow(ax_b, xs[i] + BW, 1.81, xs[i + 1], 1.81)
    _arrow(ax_b, xs[2] + BW / 2, 1.42, xs[2] + BW / 2, 0.96)
    for i in (2, 1):
        _arrow(ax_b, xs[i], 0.57, xs[i - 1] + BW, 0.57)
    _arrow(ax_b, xs[0] + BW / 2, 0.96, xs[0] + BW / 2, 1.42, ls="--", color="0.4")
    ax_b.text(xs[0] + BW / 2 + 0.10, 1.19, "一步延迟", fontsize=6.2, color="0.35", va="center")
    add_panel_label(ax_b, "b")

    align(fig, "fig1_scheme", axes=[ax_a, ax_b], panel_ids=["a", "b"],
          column_groups=[["a", "b"]])
    export(fig, "fig1_scheme")


def main() -> int:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("results", "main_metrics.csv")
    runs = load_runs(csv_path)
    have = ", ".join(sorted(f"{g}/seed{s}" for g, s in runs))
    print(f"载入 {len(runs)} 个 run：{have}")
    fig1()
    fig2(runs)
    fig3(runs)
    fig4(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
