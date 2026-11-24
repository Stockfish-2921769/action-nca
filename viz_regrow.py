#!/usr/bin/env python3
"""
可视化：再生（NCA）/ 热启动（DP）+ 边长边推（interleave，η = 0.3）遇到新障碍时，
两者成功率相同，但 NCA 的路径平滑得多。

上排：left 场景、种子 0 的闭环路径，以及沿 y 方向的平均二阶差分（路径在哪里拐折）。
下排：5 个种子的成功率，以及两个场景下“不引导 → 引导”的最大拐折。
数据：上排现场重跑 novel_obstacle.py 的同一配置；下排读 results/novel_obstacle/results.json。

用法：python viz_regrow.py   →   results/novel_obstacle/regrow_smoothness.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle

import action_nca as A
import diffusion_baseline as DB
import novel_obstacle as N

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "novel_obstacle")
NCA_C, DP_C = "#2a78d6", "#eb6834"          # 分类色第 1、2 位（蓝、橙）
INK, INK2, MUTED, GRID, SURF = "#1f1f1e", "#5f5e5a", "#8a8984", "#e6e5e0", "#fcfcfb"
ETA, SEED, E_SHOW = 0.3, 0, 60

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})


def run_paths(method, episodes=500):
    """重跑 left 场景、种子 0 的再生 / 热启动 + interleave，返回路径、状态、长度和障碍。"""
    torch.set_num_threads(4)
    circles = N.make_scene("left", episodes, np.random.default_rng(SEED + 100))
    pusher = N.Pusher(circles, ETA, 0.03)
    if method == "nca":
        m = A.ActionNCA()
        m.load_state_dict(torch.load(os.path.join(HERE, "runs", "precision", f"w1_seed{SEED}", "nca_latent.pt"),
                                     map_location="cpu"))
        planner = DB.NCAPlanner(N.GuidedNCA(m.eval(), pusher, "interleave"), "latent", "warm")
    else:
        planner = N.GuidedDP(N.load_dp(os.path.join(HERE, "runs", "novel", f"dp_seed{SEED}", "dp_policy.pt")),
                             "warm", pusher, "interleave")
    met, paths, status, length = DB.rollout(planner, episodes, 72, "cpu", SEED + 1, extra_hit=N.circle_hits(circles))
    return met, paths, status, length, circles


def max_kinks(paths, status, length):
    """每个成功回合路径上最大的二阶差分（×100）和它所在的位置。"""
    val, where = [], []
    for e in np.flatnonzero(status == 1):
        p = paths[e, :length[e] + 1]
        dd = 100 * np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1)
        val.append(dd.max())
        where.append(p[dd.argmax() + 1])
    return np.array(val), np.array(where)


def scene(ax, circles):
    ax.add_patch(Rectangle((-A.BOX_HW, -A.BOX_HH), 2 * A.BOX_HW, 2 * A.BOX_HH, fc="#d9d8d3", ec="none", zorder=1))
    x, y, r = circles[0, 0]
    ax.add_patch(Circle((x, y), r, fc="none", ec=INK2, hatch="////", lw=1.0, zorder=1))
    ax.annotate("新障碍\n（训练时没见过）", (x - r * 0.7, y + r * 0.7), xytext=(-0.98, 0.42), fontsize=7.5, color=INK2,
                ha="left", va="bottom", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.text(0, 0, "训练时的\n障碍", ha="center", va="center", fontsize=7.5, color=INK2)
    ax.plot(*A.START, "o", ms=5, color=INK, zorder=4)
    ax.plot(*A.GOAL, "*", ms=10, color=INK, zorder=4)
    ax.set(xlim=(-1.0, 1.0), ylim=(-0.95, 0.95), aspect="equal", xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_visible(False)


def paths_panel(ax, res, color, title):
    met, paths, status, length, circles = res
    scene(ax, circles)
    for e in range(E_SHOW):
        p = paths[e, :length[e] + 1]
        ax.plot(p[:, 0], p[:, 1], lw=1.0, color=color, alpha=0.55 if status[e] == 1 else 0.9, zorder=2,
                ls="-" if status[e] == 1 else ":")
    _, where = max_kinks(paths[:E_SHOW], status[:E_SHOW], length[:E_SHOW])
    ax.scatter(where[:, 0], where[:, 1], s=9, color=INK, zorder=3, lw=0)
    ax.set_title(f"{title}\n成功率 {100 * met['success']:.1f}% · 最大拐折 {met['max_kink_x100']:.1f}",
                 fontsize=9.5, loc="left")


def main():
    nca, dp = run_paths("nca"), run_paths("dp")
    rows = json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))

    def pick(method, scene_, guide, key):
        setting = "warm"
        r = sorted([x for x in rows if (x["method"], x["setting"], x["scene"], x["guide"]) == (method, setting, scene_, guide)
                    and (guide == "none" or x["eta"] == ETA)], key=lambda x: x["seed"])
        return np.array([x[key] for x in r])

    fig = plt.figure(figsize=(12.5, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.42, wspace=0.28)
    fig.suptitle("新障碍 + 边长边推（interleave，η = 0.3）：NCA 再生和 DP 热启动成功率相同，NCA 的路径平滑得多",
                 x=0.06, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.06, 0.925, "上排：left 场景、种子 0，前 60 个回合（实线成功，点线失败，黑点 = 该回合最大拐折的位置）。下排：种子 0–4，每种方式 500 回合；"
             "“拐折”= 成功回合里路径最大的二阶差分 ×100。", fontsize=8.5, color=INK2)

    paths_panel(fig.add_subplot(gs[0, 0]), nca, NCA_C, "NCA 再生 + interleave")
    paths_panel(fig.add_subplot(gs[0, 1]), dp, DP_C, "DP 热启动 + interleave")

    ax = fig.add_subplot(gs[0, 2])
    for res, c, lab in [(nca, NCA_C, "NCA 再生"), (dp, DP_C, "DP 热启动")]:
        v = np.sort(max_kinks(res[1], res[2], res[3])[0])
        q = np.arange(1, len(v) + 1) / len(v)
        ax.step(v, 100 * q, where="post", lw=2, color=c)
        med = np.median(v)
        ax.plot([med], [50], "o", ms=7, color=c, mec=SURF, mew=1.5, zorder=4)
        ax.annotate(f"{lab}  中位数 {med:.1f}", (med, 50), xytext=(8, -14 if c == DP_C else 8),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.set(xlabel="每个成功回合的最大拐折 ×100", ylabel="回合累计占比（%）", ylim=(0, 101))
    ax.set_xlim(left=0)
    ax.grid(color=GRID, lw=0.8)
    ax.set_title("最大拐折的分布（种子 0，越靠左越平滑）", fontsize=9.5, loc="left")

    seeds = np.arange(5)
    ax = fig.add_subplot(gs[1, 0])
    for i, sc in enumerate(["left", "random2"]):
        for j, (m, c) in enumerate([("nca", NCA_C), ("dp", DP_C)]):
            v = 100 * pick(m, sc, "interleave", "success")
            x = i * 3 + j + (seeds - 2) * 0.08
            ax.scatter(x, v, s=34, color=c, edgecolor=SURF, lw=1.5, zorder=3)
            ax.hlines(v.mean(), x[0] - 0.12, x[-1] + 0.12, color=INK, lw=1.5, zorder=4)
            ax.text(i * 3 + j, v.min() - 0.6, f"{v.mean():.1f}%", ha="center", va="top", fontsize=8, color=INK)
    ax.set_xticks([0, 1, 3, 4], ["NCA", "DP", "NCA", "DP"])
    for xc, sc in [(0.5, "left"), (3.5, "random2")]:
        ax.text(xc, -0.11, sc, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, color=INK)
    ax.set(ylim=(96, 100.4), ylabel="成功率（%）")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("成功率：两者持平（点 = 种子，横线 = 均值）", fontsize=9.5, loc="left")

    for k, sc in enumerate(["random2", "left"]):
        ax = fig.add_subplot(gs[1, 1 + k], sharey=None)
        for m, c, lab in [("nca", NCA_C, "NCA 再生"), ("dp", DP_C, "DP 热启动")]:
            a, b = pick(m, sc, "none", "max_kink_x100"), pick(m, sc, "interleave", "max_kink_x100")
            for s in seeds:
                ax.plot([0, 1], [a[s], b[s]], color=c, lw=1.0, alpha=0.45)
            ax.plot([0, 1], [a.mean(), b.mean()], color=c, lw=2.5, marker="o", ms=8, mec=SURF, mew=2)
            ax.annotate(f"{lab}  {a.mean():.1f} → {b.mean():.1f}（+{b.mean() - a.mean():.1f}）", (1, b.mean()),
                        xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
        ax.set_xticks([0, 1], ["不引导", "interleave η=0.3"])
        ax.set(xlim=(-0.2, 2.3), ylabel="最大拐折 ×100")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title(f"{sc}：引导带来的拐折增量（细线 = 种子）", fontsize=9.5, loc="left")

    path = os.path.join(OUT, "regrow_smoothness.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"saved {path}")
    print("seed 0 NCA:", {k: round(v, 3) for k, v in nca[0].items() if k in ("success", "max_kink_x100")},
          "DP:", {k: round(v, 3) for k, v in dp[0].items() if k in ("success", "max_kink_x100")})


if __name__ == "__main__":
    main()
