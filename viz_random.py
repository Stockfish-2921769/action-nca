#!/usr/bin/env python3
"""
可视化：障碍随机化、策略看见障碍（NOTES.md 第 17 节）。

上两排：种子 0 的模型在 5 个评估障碍上的闭环路径（3 个和训练同分布，2 个更大的 ood 障碍），
      每个障碍 12 个回合（起点各有微小扰动），第一排 NCA 再生、第二排 DP 热启动；黑点 = 每回合最大拐折的位置。
下排（种子 0–4，每种方式 500 回合）：成功率、最大拐折、计划翻转（看见障碍），以及看见 / 看不见的成功率。
数据：上两排需要 runs/random/ 里的模型（只在服务器上）；下排读 results/random_obstacle/*/metrics.json。

用法：python viz_random.py   →   results/random_obstacle/random_obstacle.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import action_nca as A
import diffusion_baseline as DB
import random_obstacle as R

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "random_obstacle")
NCA_C, DP_C = "#2a78d6", "#eb6834"          # 分类色第 1、2 位，和其他图一致
INK, INK2, GRID, SURF, OBST = "#1f1f1e", "#5f5e5a", "#e6e5e0", "#fcfcfb", "#d9d8d3"
NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"   # 服务器上的中文字体
fonts = ["Microsoft YaHei", "PingFang SC"]
if os.path.exists(NOTO):
    font_manager.fontManager.addfont(NOTO)
    fonts.insert(0, font_manager.FontProperties(fname=NOTO).get_name())
plt.rcParams.update({"font.sans-serif": fonts + ["DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
PAIRS = [("新采样", "cold_fresh", "fresh"), ("固定", "cold_fixed", "fixed"), ("再生\n热启动", "warm", "warm")]
SEEDS = range(5)


def load_models(seed=0):
    d = os.path.join(HERE, "runs", "random")
    ck = torch.load(os.path.join(d, f"nca_see_seed{seed}", "nca_latent.pt"), map_location="cpu")
    nca = A.ActionNCA(obs_dim=ck["obs_dim"])
    nca.load_state_dict(ck["state"])
    ck = torch.load(os.path.join(d, f"dp_see_seed{seed}", "dp_policy.pt"), map_location="cpu")
    dp = DB.DiffusionPolicy(ck["lo"], ck["hi"], dims=ck["dims"], k=ck["kernel"], t_dim=ck["t_dim"], groups=ck["groups"],
                            cond_lo=ck["cond_lo"], cond_hi=ck["cond_hi"])
    dp.load_state_dict(ck["state"])
    return nca.eval(), dp.eval()


def kinks(paths, status, length):
    where = []
    for e in np.flatnonzero(status == 1):
        p = paths[e, :length[e] + 1]
        dd = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1)
        where.append(p[dd.argmax() + 1])
    return np.array(where).reshape(-1, 2)


def path_panel(ax, planner, c, rect, title, E=12):
    rects = np.tile(rect[None], (E, 1)).astype(np.float32)
    cx, cy, hw, hh = rect
    ax.add_patch(Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh, fc=OBST, ec="none", zorder=1))
    m, paths, status, length = DB.rollout(planner, E, 72, "cpu", 1, rects=rects)
    for e in range(E):
        p = paths[e, :length[e] + 1]
        ok = status[e] == 1
        ax.plot(p[:, 0], p[:, 1], color=c, lw=1.0, alpha=0.6 if ok else 0.95, ls="-" if ok else ":", zorder=2)
    w = kinks(paths, status, length)
    ax.scatter(w[:, 0], w[:, 1], s=7, color=INK, lw=0, zorder=3)
    ax.plot(*A.START, "o", ms=4, color=INK, zorder=4)
    ax.plot(*A.GOAL, "*", ms=8, color=INK, zorder=4)
    ax.set(xlim=(-1.0, 1.0), ylim=(-0.95, 0.95), aspect="equal", xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{title}：成功 {int((status == 1).sum())}/{E} · 最大拐折 {m['max_kink_x100']:.1f}", fontsize=8,
                 loc="left")


def load_metrics():
    M = {}
    for m in ("nca", "dp"):
        for see in ("see", "blind"):
            for s in SEEDS:
                M[(m, see, s)] = json.load(open(os.path.join(RES, f"{m}_{see}_seed{s}", "metrics.json"), encoding="utf-8"))
    return M


def dots_panel(ax, M, key, scale, title, ylabel, ylim=None):
    xs, ticks = [], []
    for g, which in enumerate(("in", "ood")):
        for i, (lab, ns, ds) in enumerate(PAIRS):
            x0 = g * 3.6 + i * 1.1
            for j, (m, st, c) in enumerate((("nca", ns, NCA_C), ("dp", ds, DP_C))):
                v = scale * np.array([M[(m, "see", s)][f"{which}/{st}"][key] for s in SEEDS])
                x = x0 + (j - 0.5) * 0.38 + np.linspace(-0.06, 0.06, len(v))
                ax.scatter(x, v, s=14, color=c, edgecolor=SURF, lw=0.8, zorder=3)
                ax.hlines(v.mean(), x[0] - 0.07, x[-1] + 0.07, color=INK, lw=1.4, zorder=4)
            xs.append(x0)
            ticks.append(lab)
    ax.set_xticks(xs, ticks, fontsize=8)
    for g, which in enumerate(("in（同分布）", "ood（更大的障碍）")):
        ax.text(g * 3.6 + 1.1, 1.0, which, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5,
                color=INK)
    ax.axvline(2.75, color=GRID, lw=1)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set(ylabel=ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9.5, loc="left", pad=16)


def blind_panel(ax, M):
    for j, (m, c) in enumerate((("nca", NCA_C), ("dp", DP_C))):
        sts = [p[1] if m == "nca" else p[2] for p in PAIRS]
        for g, which in enumerate(("in", "ood")):
            for see, filled in (("see", True), ("blind", False)):
                v = 100 * np.array([np.mean([M[(m, see, s)][f"{which}/{st}"]["success"] for st in sts]) for s in SEEDS])
                x = g * 1.4 + j * 0.45
                ax.scatter([x] * len(v), v, s=16, facecolor=c if filled else SURF, edgecolor=c, lw=1.2, zorder=3)
                ax.hlines(v.mean(), x - 0.12, x + 0.12, color=INK, lw=1.4, zorder=4)
    ax.set_xticks([0.225, 1.625], ["in", "ood"])
    ax.set(ylim=(0, 102), ylabel="成功率（%）")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=INK2, label="看见障碍"),
                       Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, label="看不见（只给位置）")],
              loc="lower left", fontsize=7.5, frameon=False)
    ax.set_title("看见 vs 看不见（三种执行方式平均）", fontsize=9.5, loc="left", pad=16)


def main():
    nca, dp = load_models(0)
    ins = R.sample_rects(500, np.random.default_rng(1000), "in")
    ood = R.sample_rects(500, np.random.default_rng(2000), "ood")
    examples = [(ins[0], "in 1"), (ins[1], "in 2"), (ins[3], "in 3"), (ood[0], "ood 1"), (ood[2], "ood 2")]
    M = load_metrics()

    fig = plt.figure(figsize=(15, 11.6))
    gs = fig.add_gridspec(3, 20, height_ratios=[1.0, 1.0, 1.15], hspace=0.32, wspace=2.2)
    fig.suptitle("障碍随机化、策略看见障碍：NCA 和 DP 成功率打平，NCA 的路径更平滑、计划翻转更少",
                 x=0.06, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.06, 0.94, "上两排：种子 0 的模型、5 个评估障碍（前 3 个同分布，后 2 个是更大的 ood 障碍），每个障碍 12 个回合"
             "（实线成功，点线失败，黑点 = 该回合最大拐折的位置）。下排：种子 0–4，每种方式 500 回合（点 = 种子，横线 = 均值）。",
             fontsize=8.5, color=INK2)
    rows = ((DB.NCAPlanner(nca, "latent", "warm"), NCA_C, "NCA 再生"),
            (DB.DPPlanner(dp, "warm", 10, "leading", 50, 3), DP_C, "DP 热启动"))
    for row, (planner, c, name) in enumerate(rows):
        for k, (rect, title) in enumerate(examples):
            ax = fig.add_subplot(gs[row, 4 * k:4 * k + 4])
            path_panel(ax, planner, c, rect, title)
            if k == 0:
                ax.text(-0.06, 0.5, name, transform=ax.transAxes, rotation=90, ha="right", va="center",
                        fontsize=10.5, color=c, fontweight="bold")

    dots_panel(fig.add_subplot(gs[2, 0:6]), M, "success", 100, "成功率（看见障碍）", "成功率（%）", (84, 100.5))
    dots_panel(fig.add_subplot(gs[2, 6:11]), M, "max_kink_x100", 1, "最大拐折：越低越平滑", "最大拐折 ×100", (0, 11.5))
    dots_panel(fig.add_subplot(gs[2, 11:16]), M, "plan_flips", 1, "计划翻转：越低模态越连贯", "每回合计划翻转次数", (0, 1.7))
    blind_panel(fig.add_subplot(gs[2, 16:20]), M)

    fig.legend(handles=[Line2D([], [], color=NCA_C, lw=3, label="NCA（隐变量 + 精度损失）：新采样 = 每次新 z，固定 = z 固定，再生"),
                        Line2D([], [], color=DP_C, lw=3, label="DP（小模型，DDIM 10 步）：新采样 = 每次新噪声，固定 = 固定噪声，热启动")],
               loc="lower center", ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    path = os.path.join(RES, "random_obstacle.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
