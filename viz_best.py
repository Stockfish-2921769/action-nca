#!/usr/bin/env python3
"""
可视化：只推路径法向分量（--proj normal）后，综合表现最好的四种组合（NOTES.md 第 16 节）。

上排：left 场景、种子 0 的闭环路径，四种组合各一张（黑点 = 该回合最大拐折的位置）。
下排（种子 0–4，每种方式 500 回合）：
  三个场景的成功率；三个场景的最大拐折；big 场景里完整梯度 → 只推法向时成功率的变化。
数据：上排现场重跑；下排读 results/normal_push/results.json 和 results/novel_obstacle/ 里完整梯度的结果。

用法：python viz_best.py   →   results/normal_push/best_combos.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

import action_nca as A
import diffusion_baseline as DB
import novel_obstacle as N
from viz_regrow import GRID, INK, INK2, SURF, max_kinks, scene

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "normal_push")
SEED, E_SHOW = 0, 60
SCENES = ["left", "random2", "big"]
# (标签, 方法, 执行方式, 引导, η, 颜色)：分类色第 1–4 位，按固定顺序分配
COMBOS = [("NCA 每次新 z + post，η 0.1", "nca", "cold_fresh", "post", 0.1, "#2a78d6"),
          ("NCA 再生 + post，η 0.3", "nca", "warm", "post", 0.3, "#1baf7a"),
          ("DP 热启动 + post，η 0.3", "dp", "warm", "post", 0.3, "#eb6834"),
          ("DP 固定噪声 + interleave，η 0.3", "dp", "fixed", "interleave", 0.3, "#eda100")]


def run_paths(method, setting, guide, eta, episodes=500):
    torch.set_num_threads(4)
    circles = N.make_scene("left", episodes, np.random.default_rng(SEED + 100))
    pusher = N.Pusher(circles, eta, 0.03, "normal")
    if method == "nca":
        m = A.ActionNCA()
        m.load_state_dict(torch.load(os.path.join(HERE, "runs", "precision", f"w1_seed{SEED}", "nca_latent.pt"),
                                     map_location="cpu"))
        planner = DB.NCAPlanner(N.GuidedNCA(m.eval(), pusher, guide), "latent", setting)
    else:
        planner = N.GuidedDP(N.load_dp(os.path.join(HERE, "runs", "novel", f"dp_seed{SEED}", "dp_policy.pt")),
                             setting, pusher, guide)
    met, paths, status, length = DB.rollout(planner, episodes, 72, "cpu", SEED + 1, extra_hit=N.circle_hits(circles))
    return met, paths, status, length, circles


def load(path):
    return json.load(open(os.path.join(HERE, path), encoding="utf-8"))


def main():
    normal = load("results/normal_push/results.json")
    full = load("results/novel_obstacle/results.json") + load("results/novel_obstacle/big_results.json")

    def pick(rows, c, sc, key):
        _, m, st, g, e, _ = c
        r = sorted([x for x in rows if (x["method"], x["setting"], x["scene"], x["guide"]) == (m, st, sc, g)
                    and x["eta"] == e], key=lambda x: x["seed"])
        assert len(r) == 5, (c, sc, len(r))
        return np.array([x[key] for x in r])

    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.0, 0.95], hspace=0.28, wspace=1.2)
    fig.suptitle("只推路径法向分量后最好的四种组合：三个场景的成功率均值都在 99% 以上，平滑度差别明显",
                 x=0.06, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.06, 0.928, "上排：left 场景、种子 0，前 60 个回合（实线成功，点线失败，黑点 = 该回合最大拐折的位置）。"
             "下排：种子 0–4，每种方式 500 回合；“拐折”= 成功回合里路径最大的二阶差分 ×100。", fontsize=8.5, color=INK2)

    for k, c in enumerate(COMBOS):
        ax = fig.add_subplot(gs[0, 3 * k:3 * k + 3])
        met, paths, status, length, circles = run_paths(*c[1:5])
        scene(ax, circles)
        for e in range(E_SHOW):
            p = paths[e, :length[e] + 1]
            ok = status[e] == 1
            ax.plot(p[:, 0], p[:, 1], lw=1.0, color=c[5], alpha=0.55 if ok else 0.9, ls="-" if ok else ":", zorder=2)
        _, where = max_kinks(paths[:E_SHOW], status[:E_SHOW], length[:E_SHOW])
        ax.scatter(where[:, 0], where[:, 1], s=8, color=INK, zorder=3, lw=0)
        ax.set_title(f"{c[0]}\n成功率 {100 * met['success']:.1f}% · 最大拐折 {met['max_kink_x100']:.1f}",
                     fontsize=9, loc="left")
        print(c[0], round(met["success"], 3), round(met["max_kink_x100"], 2))

    xs = np.arange(len(SCENES))
    off = (np.arange(len(COMBOS)) - 1.5) * 0.2

    def dots(ax, key, scale, rows):
        for k, c in enumerate(COMBOS):
            for i, sc in enumerate(SCENES):
                v = scale * pick(rows, c, sc, key)
                x = xs[i] + off[k] + np.linspace(-0.05, 0.05, len(v))
                ax.scatter(x, v, s=16, color=c[5], edgecolor=SURF, lw=1.0, zorder=3)
                ax.hlines(v.mean(), xs[i] + off[k] - 0.08, xs[i] + off[k] + 0.08, color=INK, lw=1.5, zorder=4)
        ax.set_xticks(xs, SCENES)
        ax.grid(axis="y", color=GRID, lw=0.8)

    ax = fig.add_subplot(gs[1, 0:4])
    dots(ax, "success", 100, normal)
    ax.set(ylim=(98.5, 100.15), ylabel="成功率（%）")
    ax.set_title("成功率（点 = 种子，横线 = 均值）", fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[1, 4:8])
    dots(ax, "max_kink_x100", 1, normal)
    ax.set_ylim(bottom=0)
    ax.set(ylabel="最大拐折 ×100")
    ax.set_title("最大拐折：越低越平滑", fontsize=9.5, loc="left")

    ax = fig.add_subplot(gs[1, 8:12])
    for k, c in enumerate(COMBOS):
        a, b = 100 * pick(full, c, "big", "success"), 100 * pick(normal, c, "big", "success")
        y = len(COMBOS) - 1 - k
        ax.plot([a.mean(), b.mean()], [y, y], color=c[5], lw=2, zorder=2)
        ax.plot([a.mean()], [y], "o", ms=8, mfc=SURF, mec=c[5], mew=2, zorder=3)
        ax.plot([b.mean()], [y], "o", ms=8, color=c[5], mec=SURF, mew=1.5, zorder=3)
        to_a, to_b = 100 * pick(full, c, "big", "timeout").mean(), 100 * pick(normal, c, "big", "timeout").mean()
        ax.text(80.5, y + 0.22, c[0].split("，")[0], ha="left", va="bottom", fontsize=8.5, color=INK)
        ax.text(min(a.mean(), b.mean()) - 0.4, y, f"超时 {to_a:.1f}% → {to_b:.1f}%", ha="right", va="center",
                fontsize=8, color=INK2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set(xlim=(80, 100.5), ylim=(-1.3, len(COMBOS) - 0.2), xlabel="big 场景成功率（%）")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=SURF, mec=INK2, mew=2, label="完整梯度"),
                       Line2D([], [], marker="o", ls="", color=INK2, label="只推法向")],
              loc="lower left", ncol=2, fontsize=8, frameon=False)
    ax.set_title("big 场景（半径 0.16 的圆）：只推法向的作用", fontsize=9.5, loc="left")

    fig.legend(handles=[Line2D([], [], color=c[5], lw=3, label=c[0]) for c in COMBOS], loc="lower center",
               ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    path = os.path.join(OUT, "best_combos.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
