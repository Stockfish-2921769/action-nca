#!/usr/bin/env python3
"""
可视化：Push-T（状态版）上的 NCA 和 DP（NOTES.md 第 18 节）。

上排：种子 0 的模型在头两个测试回合（测试种子 100000、100001，没有挑选）上的闭环执行，
      NCA 再生、DP 热启动、小 DP 热启动；灰色 = 目标位姿，虚线框 = 初始 T 块，彩色 = 结束时的 T 块，
      线 = 推杆的实际轨迹，点 = 执行的动作（推杆的目标位置）。
中排左：第一个回合前 96 步的动作 x 坐标（彩色）和推杆实际 x 坐标（灰色），竖线 = 重新规划的时刻。
中排右：动作的二阶差分按“在动作块里的位置”分解（种子 0，每种方式 100 回合）：
      位置 0 = 新动作块的第一步，它的二阶差分跨过了上一个块的最后一步，所以衔接处的跳变集中在 0 和 7。
下排：种子 0–2，每种方式 100 回合（点 = 种子，横线 = 均值）。
数据：上排和中排要 runs/pusht/ 里的模型和 gym_pusht（只在服务器上）；下排读 results/pusht/*/metrics.json。

用法：python viz_pusht.py   →   results/pusht/pusht.png（中排的分解数字另存 results/pusht/chunk_position.json）
"""
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

import pusht as P

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "pusht")
RUNS = os.path.join(HERE, "runs", "pusht")
METHODS = ("nca", "dp", "dp_small")
COL = {"nca": "#2a78d6", "dp": "#eb6834", "dp_small": "#1baf7a"}   # 分类色第 1、2、3 位，和其他图一致
NAME = {"nca": "NCA", "dp": "DP", "dp_small": "小 DP"}
INK, INK2, GRID, SURF, GOAL = "#1f1f1e", "#5f5e5a", "#e6e5e0", "#fcfcfb", "#d9d8d3"
NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"   # 服务器上的中文字体
fonts = ["Microsoft YaHei", "PingFang SC"]
if os.path.exists(NOTO):
    font_manager.fontManager.addfont(NOTO)
    fonts.insert(0, font_manager.FontProperties(fname=NOTO).get_name())
plt.rcParams.update({"font.sans-serif": fonts + ["DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
# (标签, NCA 的执行方式, DP 的执行方式)
PAIRS = [("新采样", "fresh", "fresh"), ("固定", "fixed", "fixed"), ("再生 /\n热启动", "warm", "warm")]
SEEDS = range(3)

# T 块的两个矩形（gym_pusht 的 add_tee，scale = 30，length = 4），块坐标系
S, L = 30, 4
T_SHAPES = [np.array([(-L * S / 2, S), (L * S / 2, S), (L * S / 2, 0), (-L * S / 2, 0)], float),
            np.array([(-S / 2, S), (-S / 2, L * S), (S / 2, L * S), (S / 2, S)], float)]
GOAL_POSE = (256.0, 256.0, np.pi / 4)


def tee(ax, pose, **kw):
    x, y, th = pose
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for v in T_SHAPES:
        ax.add_patch(Polygon(v @ R.T + (x, y), closed=True, **kw))


def kind(m):
    return P.VARIANTS[m][0]


def planner(m, mode, seed=0):
    k = kind(m)
    return P.make_planner(k, os.path.join(RUNS, f"{m}_seed{seed}", f"{k}.pt"), mode)


def board(ax, tr, e, m):
    ax.add_patch(Polygon([(0, 0), (512, 0), (512, 512), (0, 512)], closed=True, fc="white", ec=GRID, lw=0.8))
    tee(ax, GOAL_POSE, fc=GOAL, ec="none", zorder=1)
    tee(ax, tr["block"][e][0], fill=False, ec=INK2, lw=0.9, ls=(0, (3, 2)), zorder=2)
    tee(ax, tr["block"][e][-1], fc=COL[m], ec=COL[m], alpha=0.35, lw=1.0, zorder=3)
    ag, ac = tr["agent"][e], tr["acts"][e]
    ax.plot(ag[:, 0], ag[:, 1], color=COL[m], lw=1.1, zorder=4)
    ax.scatter(ac[:, 0], ac[:, 1], s=2.5, color=INK, lw=0, alpha=0.55, zorder=5)
    ax.plot(*ag[0], "o", ms=5, mfc="white", mec=INK, mew=1.0, zorder=6)
    ax.set(xlim=(0, 512), ylim=(512, 0), aspect="equal", xticks=[], yticks=[])   # y 向下，和环境渲染的画面一致
    for s in ax.spines.values():
        s.set_visible(False)
    ok = "成功" if tr["success"][e] else "未成功"
    ax.set_title(f"{NAME[m]}：得分 {tr['best'][e]:.2f}，{ok}，{len(ac)} 步", fontsize=8, loc="left")


def chunk_profile(acts):
    """动作二阶差分（像素）按在动作块里的位置 0–7 求平均。"""
    num, den = np.zeros(P.T_A), np.zeros(P.T_A)
    for a in acts:
        if len(a) < 3:
            continue
        d = np.linalg.norm(a[2:] - 2 * a[1:-1] + a[:-2], axis=1)   # d[j] 对应第 j + 1 步
        pos = (np.arange(1, len(a) - 1)) % P.T_A
        np.add.at(num, pos, d)
        np.add.at(den, pos, 1)
    return num / den


def load_metrics():
    return {(m, s): json.load(open(os.path.join(RES, f"{m}_seed{s}", "metrics.json"), encoding="utf-8"))
            for m in METHODS for s in SEEDS}


def demo_rough():
    """人类示范本身的动作 / 推杆粗糙度（像素，逐回合平均），作为参考线。"""
    import zarr
    z = zarr.open(P.DATA, "r")
    act, st, ends = z["data/action"][:], z["data/state"][:, :2], z["meta/episode_ends"][:]
    r = lambda q: np.linalg.norm(np.diff(q, 2, axis=0), axis=1).mean()
    segs = list(zip(np.r_[0, ends[:-1]], ends))
    return (float(np.mean([r(act[a:b]) for a, b in segs if b - a > 2])),
            float(np.mean([r(st[a:b]) for a, b in segs if b - a > 2])))


def dots_panel(ax, M, key, scale, title, ylabel, ylim=None, log=False, ref=None):
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(1.0, ref, f"人类示范 {ref:.1f}", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=7.5, color=INK2)
    xs = []
    for i, (lab, ns, ds) in enumerate(PAIRS):
        x0 = i * 1.3
        for j, m in enumerate(METHODS):
            st = ns if kind(m) == "nca" else ds
            v = scale * np.array([M[(m, s)][st][key] for s in SEEDS])
            x = x0 + (j - 1) * 0.34 + np.linspace(-0.05, 0.05, len(v))
            ax.scatter(x, v, s=14, color=COL[m], edgecolor=SURF, lw=0.8, zorder=3)
            ax.hlines(v.mean(), x[0] - 0.06, x[-1] + 0.06, color=INK, lw=1.4, zorder=4)
        xs.append(x0)
    ax.set_xticks(xs, [p[0] for p in PAIRS], fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    if log:
        ax.set_yscale("log")
        ax.set_yticks([20, 50, 100, 200, 500, 1000], ["20", "50", "100", "200", "500", "1000"])
        ax.minorticks_off()
    ax.set(ylabel=ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9.5, loc="left")


def main():
    torch.set_num_threads(1)                         # 和 pusht.eval_setting 一样单线程，结果应和 metrics.json 种子 0 一致
    M = load_metrics()
    # 种子 0 的模型，每种方式 100 个测试回合（和 metrics.json 里的评估同样的测试种子，rollout 的随机种子 = 1）
    cache = os.path.join(RUNS, "viz_rollouts.pkl")   # 只改图时不用重跑（删掉它就重跑）
    if os.path.exists(cache):
        T = pickle.load(open(cache, "rb"))
    else:
        T = {}
        for m in METHODS:
            for _, st, _ in PAIRS:
                r, T[(m, st)] = P.rollout(planner(m, st), 100, 1)
                print(f"rollout {m:8s} {st:5s} score {r['score']:.3f}（metrics.json {M[(m, 0)][st]['score']:.3f}）"
                      f"  action_rough {r['action_rough']:.3f}（{M[(m, 0)][st]['action_rough']:.3f}）", flush=True)
        pickle.dump(T, open(cache, "wb"))
    prof = {f"{m}/{st}": chunk_profile(T[(m, st)]["acts"]).round(3).tolist() for (m, st) in T}
    with open(os.path.join(RES, "chunk_position.json"), "w", encoding="utf-8") as f:
        json.dump(prof, f, indent=1, ensure_ascii=False)

    demo_act, demo_agent = demo_rough()
    fig = plt.figure(figsize=(16, 11.6))
    gs = fig.add_gridspec(3, 18, height_ratios=[0.66, 0.9, 0.95], hspace=0.31, wspace=2.4, top=0.91)
    fig.suptitle(TITLE, x=0.06, y=0.998, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.06, 0.962, SUBTITLE, fontsize=8.5, color=INK2)

    # 上排：两个测试回合 × 三种方法
    for e in range(2):
        for j, m in enumerate(METHODS):
            ax = fig.add_subplot(gs[0, 9 * e + 3 * j:9 * e + 3 * j + 3])
            board(ax, T[(m, "warm")], e, m)
            if j == 0:
                ax.text(0.0, 1.13, f"测试回合 {e + 1}（测试种子 {P.TEST_SEED + e}），都是再生 / 热启动", transform=ax.transAxes,
                        fontsize=9, color=INK, ha="left")

    # 中排左：动作 x 坐标随时间
    sub = gs[1, 0:11].subgridspec(3, 1, hspace=0.25)
    for j, m in enumerate(METHODS):
        ax = fig.add_subplot(sub[j])
        tr = T[(m, "warm")]
        n = min(96, len(tr["acts"][0]))
        for t in range(0, n + 1, P.T_A):
            ax.axvline(t, color=GRID, lw=0.8, zorder=0)
        ax.plot(np.arange(n + 1), tr["agent"][0][:n + 1, 0], color="#a8a7a1", lw=1.6, zorder=1)
        ax.plot(np.arange(n), tr["acts"][0][:n, 0], color=COL[m], lw=1.1, marker="o", ms=1.8, zorder=2)
        ax.set_xlim(0, 96)
        ax.text(0.005, 0.92, f"{NAME[m]}（{'再生' if kind(m) == 'nca' else '热启动'}）", transform=ax.transAxes,
                fontsize=8, color=COL[m], va="top", fontweight="bold", zorder=5,
                bbox=dict(fc=SURF, ec="none", pad=1.5))
        ax.set_ylabel("x（像素）", fontsize=8)
        if j < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("步数（竖线 = 重新规划）")
        if j == 0:
            ax.set_title("测试回合 1 前 96 步：动作 x 坐标（彩色）和推杆实际 x 坐标（灰色）", fontsize=9.5, loc="left")

    # 中排右：二阶差分按块内位置分解
    ax = fig.add_subplot(gs[1, 12:18])
    for m in METHODS:
        for st, ls, mk in (("fixed", (0, (3, 2)), "o"), ("warm", "-", "o")):
            p = prof[f"{m}/{st}"]
            ax.plot(range(P.T_A), p, color=COL[m], ls=ls, lw=1.6 if st == "warm" else 1.1, marker=mk,
                    ms=4 if st == "warm" else 3, mfc=COL[m] if st == "warm" else SURF, mec=COL[m], zorder=3)
    ax.set_xticks(range(P.T_A), ["0\n新块第一步"] + [str(i) for i in range(1, P.T_A - 1)] + ["7\n块末"], fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylabel("动作二阶差分（像素）")
    ax.set_ylim(bottom=0)
    ax.legend(handles=[Line2D([], [], color=INK2, lw=1.6, marker="o", ms=4, label="再生 / 热启动"),
                       Line2D([], [], color=INK2, lw=1.1, ls=(0, (3, 2)), marker="o", ms=3, mfc=SURF, label="固定")],
              loc="upper center", fontsize=7.5, frameon=False, ncol=2)
    ax.set_title("动作抖动在块内的位置（种子 0，每种方式 100 回合）", fontsize=9.5, loc="left")

    # 下排：逐种子指标
    panels = [("score", 1, "得分（覆盖率奖励最大值）", "得分", None, False, None),
              ("success", 100, "成功率（覆盖率 > 0.95）", "成功率（%）", (0, 100), False, None),
              ("action_rough", 1, "动作粗糙度：越低越平滑", "动作二阶差分（像素）", None, False, demo_act),
              ("agent_rough", 1, "推杆实际轨迹粗糙度", "推杆二阶差分（像素）", None, False, demo_agent),
              ("chunk_jump", 1, "块间衔接跳变（像素）", None, None, False, None),
              ("ms_per_plan", 1, "规划耗时（毫秒，对数轴）", None, None, True, None)]
    for i, (key, sc, title, yl, lim, log, ref) in enumerate(panels):
        dots_panel(fig.add_subplot(gs[2, 3 * i:3 * i + 3]), M, key, sc, title, yl, lim, log, ref)

    fig.legend(handles=[Line2D([], [], color=COL[m], lw=3, label=LEGEND[m]) for m in METHODS],
               loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    path = os.path.join(RES, "pusht.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")


TITLE = "Push-T（状态版）：NCA 冷启动的成功率和 DP 打平，但动作更抖、得分更低；再生反而掉到 33%"
SUBTITLE = ("上排：种子 0 的模型、头两个测试回合（没有挑选）；灰色 = 目标位姿，虚线框 = 初始 T 块，彩色 = 结束时的 T 块，"
            "线 = 推杆轨迹，黑点 = 执行的动作。下排：种子 0–2，每种方式 100 回合（点 = 种子，横线 = 均值，虚线 = 人类示范本身）。")
LEGEND = {"nca": "NCA（12 万参数）：新采样 = 每次新 z，固定 = z 固定，再生",
          "dp": "DP（413 万参数，DDIM 10 步）：新噪声 / 固定噪声 / 热启动",
          "dp_small": "小 DP（12.5 万参数）：同上"}

if __name__ == "__main__":
    main()
