#!/usr/bin/env python3
"""
可视化：Push-T 上 NCA 的调参（NOTES.md 第 19 节）。

左上：每个训练变体的彻底失败率（每次新 z 冷启动；点 = 训练种子，竖线 = 均值；◇ = 每个种子 500 回合的复评），
      小 DP、DP 作参照（虚线 = 500 回合复评的均值）；“+ DAgger”“+ DART”是第七、八轮补数据后重训的版本（小 DP 的只有 500 回合复评）。
右上：卡住的状态上，计划第 k 步离当前推杆位置的距离：NCA（默认冷启动 32 步）、同一批状态上的 DP 和小 DP；
      灰色虚线是 NCA 在正常状态上的计划。
左下：吸引子：卡住的状态上，NCA 从“停在原地”（默认冷启动）或从 DP 的计划出发，每长一步后计划第 8 步离推杆多远；
      灰色是正常状态。
右下：评估时冷启动的生长步数（不重训；se、se_zemb、se_exec 各 3 个种子）。
线 = 所有（模型, 种子）的中位数，色带 = 最小到最大。

第二张图（有 dagger_fit.json 时画 dagger.png，有 dart_fit.json 时画 dart.png，NOTES.md 19.8、19.9）：补数据前后。
左：彻底失败率（500 回合 × 3 个种子）；右：计划第 8 步离推杆多远，在补的数据里的状态（训练时见过）、
DART 时还有学生自己卡住的状态（没见过），以及测试时卡住的状态上。dart.png 下面一行是接管检验（有 takeover.json 时）：
测试时卡住的那一刻换自己（新随机数）、大 DP、小 DP 接手，脱离彻底失败的比例。

第三张图（有 rescue.json 和 rescue_fit.json 时画 rescue.png，NOTES.md 19.10）：按结果挑标签。左上：彻底失败率和得分
随补数据的轮次（只用示范 → + DART → + 按结果挑标签）；左中：学生卡住的状态里，每个接手的规划器能救出多少；
右：四组卡住的状态上补数据前后的计划第 8 步；下面：接管检验（DART 后和按结果挑标签后的模型）。
第十轮（再收集一轮，NOTES.md 19.11）的 rescue2.png 是同样的四块：学生换成第九轮的模型，接手者多了第九轮的两个模型，
轮次多了一轮；第十轮的模型 3 个种子都有 rescue_fit 结果时才画。

第四张图（有 paths.json 时）：闭环路径。行 = DAgger 前后的 NCA 和小 DP，列 = 按规则取的测试回合（同一列初始状态相同）；
彻底失败的格子里标出第一次卡住的位置，以及那一刻模型自己和大 DP 在同一观测上的计划。

第五张图（有 big.json 时画 big.png，NOTES.md 19.11）：大样本对比。上：第九、十轮的 NCA 和小 DP 的得分、成功率、
彻底失败率（每个种子 8000 个新回合，和 500 个测试回合的均值并排）；下：同样回合上的配对差和 95% 区间。

数据：runs/pusht_tune/ 下的 cat.json、attract.json、steps.json、diag.json、final.json、dagger.json、dagger_fit.json、
dart_fit.json、takeover.json、rescue.json、rescue_fit.json、paths.json、big.json（pusht_tune.py 生成）；没有 runs/ 时读
results/pusht_tune/。

用法：python viz_tune.py   →   results/pusht_tune/tune.png、dagger.png、dart.png、rescue.png、rescue2.png、paths.png、big.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "runs", "pusht_tune")
if not os.path.exists(os.path.join(SRC, "cat.json")):
    SRC = os.path.join(HERE, "results", "pusht_tune")
OUT = os.path.join(HERE, "results", "pusht_tune", "tune.png")
DAG_OUT = os.path.join(HERE, "results", "pusht_tune", "dagger.png")
PATHS_OUT = os.path.join(HERE, "results", "pusht_tune", "paths.png")

COL = {"nca": "#2a78d6", "dp": "#eb6834", "dp_small": "#1baf7a"}   # 和 viz_pusht.py 一致
INK, INK2, MUTED, GRID, SURF = "#1f1f1e", "#5f5e5a", "#a8a7a1", "#e6e5e0", "#fcfcfb"
NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"   # 服务器上的中文字体
fonts = ["Microsoft YaHei", "PingFang SC"]
if os.path.exists(NOTO):
    font_manager.fontManager.addfont(NOTO)
    fonts.insert(0, font_manager.FontProperties(fname=NOTO).get_name())
plt.rcParams.update({"font.sans-serif": fonts + ["DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})

# 变体按轮次排列（NOTES.md 19.1 的表），名字 → 显示的标签
VARIANTS = [("nca", "基线"), ("sync", "sync"), ("ema", "ema"), ("smooth", "smooth"), ("big", "big"), ("rnoise", "rnoise"),
            ("se", "se"), ("se_noise", "se_noise"), ("se_k16", "se_k16"), ("se_long", "se_long"),
            ("se_persist", "se_persist"), ("se_prand", "se_prand"), ("se_pp", "se_pp"), ("se_jit4", "se_jit4"),
            ("se_jit8", "se_jit8"), ("se_zemb", "se_zemb"), ("se_exec", "se_exec"), ("se_global", "se_global"),
            ("se_nopad", "se_nopad"), ("se_global_dag", "se_global + DAgger"), ("se_global_dart", "se_global + DART"),
            ("se_global_rescue", "se_global + 按结果挑标签"), ("se_global_rescue2", "se_global + 再收集一轮")]


def load(name):
    return json.load(open(os.path.join(SRC, name), encoding="utf-8"))


def band(ax, x, curves, color, ls="-", lw=1.8, label=None, alpha=0.12):
    """所有（模型, 种子）曲线的中位数 + 最小到最大的色带。"""
    c = np.array(curves, float)
    ax.fill_between(x, c.min(0), c.max(0), color=color, alpha=alpha, lw=0, zorder=1)
    ax.plot(x, np.median(c, 0), color=color, ls=ls, lw=lw, zorder=3, label=label)
    return np.median(c, 0)


def panel_cat(ax, cat, diag, final):
    rows = [(v, lab, [cat[k]["catastrophic"] * 100 for k in sorted(cat) if k.split("/")[0] == v]) for v, lab in VARIANTS]
    rows = [r for r in rows if r[2]]
    ref = {}
    for r in diag:                                                         # 小 DP、DP：diag.json 的逐回合最好覆盖率
        if r["model"] in ("dp", "dp_small"):
            ref.setdefault(r["model"], []).append(float(np.mean(np.array(r["best"]) < 0.5)) * 100)
    big = {}                                                               # 500 回合复评（final.json，每次新 z / 新噪声）
    for k, v in final.items():
        m, mode, _ = k.split("/")
        if mode == "fresh":
            big.setdefault(m, []).append(v["catastrophic"] * 100)
    rows += [("dp_small", "小 DP", ref["dp_small"])]
    rows += [(m, lab, []) for m, lab in (("dp_small_dag", "小 DP + DAgger"), ("dp_small_dart", "小 DP + DART"),
                                         ("dp_small_rescue", "小 DP + 按结果挑标签"), ("dp_small_rescue2", "小 DP + 再收集一轮"))
             if m in big]                                                  # 只有 500 回合复评，没有 100 回合的点
    rows += [("dp", "DP", ref["dp"])]
    ys = np.arange(len(rows))[::-1]
    for y, (v, lab, vals) in zip(ys, rows):
        col = COL["dp_small"] if v.startswith("dp_small") else COL.get(v, COL["nca"])
        ax.scatter(vals, [y] * len(vals), s=16, color=col, edgecolor=SURF, lw=0.8, zorder=3,
                   alpha=1.0 if len(vals) > 1 or v in COL else 0.55)
        if len(vals) > 1:
            ax.vlines(np.mean(vals), y - 0.32, y + 0.32, color=INK, lw=1.4, zorder=4)
        if v in big:
            ax.scatter([np.mean(big[v])], [y], s=46, marker="D", facecolor="none", edgecolor=INK, lw=1.1, zorder=5)
    for v in ("dp_small", "dp"):
        ax.axvline(np.mean(big.get(v, ref[v])), color=COL[v], lw=1.0, ls=(0, (4, 3)), zorder=0)
    ax.set_yticks(ys, [f"{lab}（{len(vals)}）" if vals else lab for _, lab, vals in rows], fontsize=7.5)
    ax.set_xlim(0, 16)
    ax.set_xlabel("彻底失败率（%）；点 = 种子（100 回合），竖线 = 均值，浅色点 = 只有 1 个种子，◇ = 500 回合 × 3 个种子的均值",
                  fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    nca_big = [np.mean(x) for m, x in big.items() if not m.startswith("dp")]
    title = (f"彻底失败率：NCA {min(nca_big):.1f}–{max(nca_big):.1f}%，小 DP {np.mean(big['dp_small']):.1f}%，"
             f"DP {np.mean(big['dp']):.1f}%（500 回合）" if nca_big and "dp_small" in big and "dp" in big else
             f"彻底失败率：小 DP {np.mean(ref['dp_small']):.1f}%，DP {np.mean(ref['dp']):.1f}%")
    ax.set_title(title, fontsize=9.5, loc="left")


def panel_profile(ax, att):
    x = np.arange(1, 17)
    st = [v["stall"] for v in att.values() if "stall" in v and "prof_nca" in v["stall"]]
    nm = [v["normal"] for v in att.values() if "normal" in v and "prof_nca" in v["normal"]]
    m_nm = band(ax, x, [r["prof_nca"] for r in nm], MUTED, ls=(0, (4, 3)), lw=1.4, alpha=0.0)
    m_nca = band(ax, x, [r["prof_nca"] for r in st], COL["nca"])
    m_dps = band(ax, x, [r["prof_dp_small"] for r in st], COL["dp_small"], alpha=0.0)
    m_dp = band(ax, x, [r["prof_dp"] for r in st], COL["dp"], alpha=0.0)
    ends = sorted([(m_nca[-1], "NCA", COL["nca"]), (m_dps[-1], "小 DP", COL["dp_small"]), (m_dp[-1], "DP", COL["dp"]),
                   (m_nm[-1], "NCA，正常状态", INK2)])
    y_prev = -1e9
    for y, lab, col in ends:                                              # 线末标签，至少隔 7 像素，避免重叠
        y = max(y, y_prev + 7)
        ax.text(16.3, y, lab, fontsize=8, color=col, va="center")
        y_prev = y
    ax.axvspan(0.5, 8.5, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.text(4.5, 2, "会执行的前 8 步", fontsize=7.5, color=INK2, ha="center")
    ax.set_xlim(0.5, 19.5)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_ylim(0, None)
    ax.set_xlabel("计划第 k 步")
    ax.set_ylabel("离当前推杆位置（像素，中位数）")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title(f"卡住的状态上：NCA 前 8 步几乎不动，DP 和小 DP 照常走（{len(st)} 个模型）", fontsize=9.5, loc="left")


def panel_attract(ax, att):
    x = np.arange(1, 33)
    st = [v["stall"] for v in att.values() if "stall" in v and "blank_curve" in v["stall"]]
    nm = [v["normal"] for v in att.values() if "normal" in v and "blank_curve" in v["normal"]]
    lines = [(nm, "blank_curve", MUTED, "-", "正常状态，从停在原地出发"), (nm, "from_dp_curve", MUTED, (0, (4, 3)), "正常状态，从 DP 计划出发"),
             (st, "blank_curve", COL["nca"], "-", "卡住的状态，从停在原地出发"),
             (st, "from_dp_curve", COL["nca"], (0, (4, 3)), "卡住的状态，从 DP 计划出发")]
    for rs, key, col, ls, lab in lines:
        band(ax, x, [r[key] for r in rs], col, ls=ls, label=lab, alpha=0.10 if col == COL["nca"] else 0.0)
    ax.axvline(32, color=INK2, lw=0.8, ls=":", zorder=0)
    ax.text(31.6, 36, "评估时\n长 32 步", fontsize=7, color=INK2, ha="right")
    ax.set_xlim(1, 33)
    ax.set_xticks([1, 8, 16, 24, 32])
    ax.set_ylim(0, None)
    ax.set_xlabel("NCA 生长步数")
    ax.set_ylabel("计划第 8 步离推杆（像素，中位数）")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    ax.set_title("吸引子：卡住的状态上，从 DP 的好计划出发也会被拉回原地", fontsize=9.5, loc="left")


def panel_steps(ax, steps, cat_ref):
    grid = sorted({int(k.split("/")[1]) for k in steps})
    keys = sorted({(k.split("/")[0], k.split("/")[2]) for k in steps})
    for metric, col, lab in (("catastrophic", COL["nca"], "彻底失败率"), ("success", INK2, "成功率")):
        curves = [[steps[f"{m}/{c}/{s}"][metric] * 100 for c in grid] for m, s in keys
                  if all(f"{m}/{c}/{s}" in steps for c in grid)]
        med = band(ax, grid, curves, col, alpha=0.12)
        ax.scatter(grid, med, s=18, color=col, edgecolor=SURF, lw=0.8, zorder=4)
        ax.text(grid[-1] + 0.6, med[-1], lab, fontsize=8, color=col, va="center")
    ax.axhline(cat_ref, color=COL["dp_small"], lw=1.0, ls=(0, (4, 3)), zorder=0)
    ax.text(12.3, cat_ref + 1.2, f"小 DP 的彻底失败率 {cat_ref:.1f}%", fontsize=7.5, color=COL["dp_small"])
    ax.set_xticks(grid)
    ax.set_xlim(grid[0] - 1, grid[-1] + 5)
    ax.set_ylim(0, 75)
    ax.set_xlabel("评估时冷启动的生长步数（训练时 24–40 步）")
    ax.set_ylabel("占回合数（%）")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title(f"少长几步不能避开吸引子：步数越少，彻底失败越多（{len(keys)} 个模型）", fontsize=9.5, loc="left")


# 补数据前后的图：键 → 后缀、图里的名字、输出、右图的状态组（dagger_fit.json / dart_fit.json 里 fit 的子集）和文字
EXTRA_FIGS = {
    "dag": dict(suffix="_dag", what="DAgger", out=DAG_OUT,
                groups=[("cat", "DAgger 数据里卡住的状态（训练时见过，标签 = 同种子大 DP 的计划）")],
                title="DAgger 补数据（一轮）：NCA 学会了标签，但彻底失败率没有下降；小 DP 反而变差",
                right_title="见过的卡住状态学会了走；测试时卡在没见过的地方，照样不动",
                note="NCA = se_global。学生每次新 z / 新噪声闭环 500 个回合（环境种子 200000 起），"
                     "每次规划时的观测用同种子大 DP 的计划做标签，和示范一起从头重训。"),
    "dart": dict(suffix="_dart", what="DART", out=os.path.join(HERE, "results", "pusht_tune", "dart.png"),
                 groups=[("far", "DART 数据里离示范远的状态（训练时见过，标签 = 老师扰动前的计划）"),
                         ("cat", "学生自己卡住的状态（DAgger 数据；DART 后的模型没见过）")],
                 title="DART 补数据：NCA 推广到了没见过的卡住状态，彻底失败略降；剩下的卡在老师自己也出不来的地方",
                 right_title="没见过的卡住状态也学会了走（推广成立）；测试时卡住的地方照样不动",
                 take_title="卡住时换人接手：NCA 原来卡住的地方，老师能救出 68%；DART 后卡住的地方，老师只能救出 10%",
                 note="NCA = se_global。老师（同种子大 DP）自己闭环，执行前给动作块加扰动（整体平移、按一定概率放慢），"
                      "扰动前的计划作标签；NCA 和小 DP 用同一份数据从头重训，训练轮数随数据量放大。\n"
                      "接管：最终对比的测试回合里第一次卡住时，重放到那个状态，换人接手跑到第 300 步（下图大点 = 3 个种子合起来）。"),
}


TAKE_SRC = [("se_global", "NCA"), ("se_global_dag", "NCA + DAgger"), ("se_global_dart", "NCA + DART"),
            ("dp_small", "小 DP"), ("dp_small_dag", "小 DP + DAgger"), ("dp_small_dart", "小 DP + DART")]
TAKE_CTRL = [("self", "卡住的模型自己（换随机数）"), ("dp", "大 DP（老师）"), ("dp_small", "小 DP（只用示范）")]


def panel_takeover(ax, take, title, srcs=TAKE_SRC):
    """接管检验（takeover.json）：每行一个卡住的来源（srcs），点 = 换谁接手后脱离彻底失败的比例；小点 = 种子，大点 = 3 个种子合起来。"""
    rows = [(m, lab) for m, lab in srcs if any(k.split("/")[0] == m and v["n"] for k, v in take.items())]
    ys = np.arange(len(rows))[::-1].astype(float)
    ys[[i for i, (m, _) in enumerate(rows) if m.startswith("dp_small")]] -= 0.5    # NCA 和小 DP 两组之间留空
    ticks = []
    for y, (m, lab) in zip(ys, rows):
        outs = [take[f"{m}/{s}"] for s in "012" if f"{m}/{s}" in take and take[f"{m}/{s}"]["n"]]
        src = COL["dp_small"] if m.startswith("dp_small") else COL["nca"]
        pooled = {}
        for c, _ in TAKE_CTRL:
            per = [100 * np.mean(np.array(o[c]["best"]) >= 0.5) for o in outs if c in o]
            if not per:
                continue
            pooled[c] = 100 * np.mean(np.concatenate([np.array(o[c]["best"]) >= 0.5 for o in outs if c in o]))
            style = dict(facecolor=SURF, edgecolor=src, lw=1.5) if c == "self" else dict(color=COL[c], edgecolor=SURF, lw=0.9)
            ax.scatter(per, [y] * len(per), s=13, alpha=0.6, zorder=3, **{**style, "lw": 0.8})
            ax.scatter([pooled[c]], [y], s=64, zorder=5, **style)
        ax.hlines(y, min(pooled.values()), max(pooled.values()), color=GRID, lw=3, zorder=1)
        ticks.append(f"{lab}（{sum(o['n'] for o in outs)}）")
    ax.set_yticks(ys, ticks, fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(ys.min() - 0.6, ys.max() + 0.6)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("接手后脱离彻底失败（整回合最好覆盖率 ≥ 0.5）的比例（%）；括号里是 3 个种子合起来卡住的回合数")
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=7.5, mfc=SURF, mec=INK2, mew=1.4, label=TAKE_CTRL[0][1]),
                       Line2D([], [], ls="none", marker="o", ms=7.5, mfc=COL["dp"], mec=COL["dp"], label=TAKE_CTRL[1][1]),
                       Line2D([], [], ls="none", marker="o", ms=7.5, mfc=COL["dp_small"], mec=COL["dp_small"],
                              label=TAKE_CTRL[2][1])],
              loc="center right", fontsize=8, frameon=False)                         # 右边 75% 以上没有点
    ax.set_title(title, fontsize=9.5, loc="left")


def dagger_figure(final, fit, cover, dag, key="dag", take=None):
    """补数据前后（se_global、小 DP 各 3 个种子）。fit / cover 是 dagger_fit.json（dart_fit.json）的两部分，dag 是 dagger.json；
    take 是 takeover.json（有的话在下面加一行：卡住的时候换人接手）。"""
    cfg = EXTRA_FIGS[key]
    sfx, what, groups = cfg["suffix"], cfg["what"], cfg["groups"]
    seeds = "012"
    models = [("se_global", "NCA", COL["nca"]), ("dp_small", "小 DP", COL["dp_small"])]
    color = {m: c for m, _, c in models}
    H0 = 4.9 + 1.3 * (len(groups) - 1)                                    # 右图每多一组状态，图高 1.3 英寸
    HT = 3.3 if take else 0.0                                             # 接管检验那一行
    H = H0 + HT
    fig = plt.figure(figsize=(12.5, H))
    gs = fig.add_gridspec(2 if take else 1, 2, width_ratios=[1, 1.55], wspace=0.32, hspace=0.35 if take else 0.2,
                          height_ratios=[H0 - 1.18, HT - 0.9] if take else None)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(top=1 - 1.18 / H)
    if take:
        panel_takeover(fig.add_subplot(gs[1, :]), take, cfg["take_title"])

    # 左：彻底失败率，细线 = 种子，粗线 = 均值
    ends, title = [], []
    for m, lab, col in models:
        b = [final[f"{m}/fresh/{s}"]["catastrophic"] * 100 for s in seeds]
        a = [final[f"{m}{sfx}/fresh/{s}"]["catastrophic"] * 100 for s in seeds]
        for x0, x1 in zip(b, a):
            a1.plot([0, 1], [x0, x1], color=col, lw=0.9, alpha=0.35, zorder=2)
        a1.plot([0, 1], [np.mean(b), np.mean(a)], color=col, lw=2.2, marker="o", ms=6, mec=SURF, mew=1, zorder=3)
        a1.text(-0.07, np.mean(b), f"{np.mean(b):.1f}%", color=INK2, fontsize=8.5, ha="right", va="center")
        ends.append([np.mean(a), f"{lab} {np.mean(a):.1f}%", col])
        title.append(f"{lab} {np.mean(b):.1f}% → {np.mean(a):.1f}%")
    ends.sort()
    for i in range(1, len(ends)):                                          # 线末标签至少隔 0.8 个百分点，避免重叠
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.8)
    for y, t, col in ends:
        a1.scatter([1.06], [y], marker="s", s=22, color=col, lw=0, zorder=4)
        a1.text(1.1, y, t, color=INK2, fontsize=8.5, va="center")
    dp = np.mean([final[f"dp/fresh/{s}"]["catastrophic"] * 100 for s in seeds])
    a1.axhline(dp, color=COL["dp"], lw=1.0, ls=(0, (4, 3)), zorder=0)
    a1.text(0.5, dp - 0.35, f"老师（大 DP）{dp:.1f}%", color=INK2, fontsize=7.5, ha="center", va="top")
    a1.set_xticks([0, 1], [f"{what} 前", f"{what} 后"])
    a1.set_xlim(-0.4, 1.95)
    a1.set_ylim(0, 12)
    a1.set_ylabel("彻底失败率（%，500 回合）")
    a1.grid(axis="y", color=GRID, lw=0.8)
    a1.set_title("彻底失败率：" + "，".join(title), fontsize=9.5, loc="left")

    # 右：计划第 8 步。空心圆 = 补数据前，实心圆 = 补数据后，空心菱形 = 大 DP 的标签；小点 = 种子，大点 = 均值。
    # 每组状态一个小标题、两行（NCA、小 DP），最后一组是测试时卡住的状态；组的标题从下往上每隔 2.6 放一个
    heads = [2.02 + 2.6 * (len(groups) - k) for k in range(len(groups))]
    rows = {}
    for (g, text), yh in zip(groups, heads):
        a2.text(0.5, yh, text, fontsize=8.5, color=INK)
        a2.axhline(yh - 2.32, color=GRID, lw=0.8)
        rows.update({(g, "se_global"): yh - 0.62, (g, "dp_small"): yh - 1.42})
    rows.update({("cover", "se_global"): 1.4, ("cover", "dp_small"): 0.6})
    for (g, m), y in rows.items():
        col = color[m]
        if g != "cover":
            b = [fit[f"{m}/{s}"][g]["before"]["d8"] for s in seeds]
            a = [fit[f"{m}/{s}"][g]["after"]["d8"] for s in seeds]
            lab = [fit[f"{m}/{s}"][g]["label8"] for s in seeds]
            a2.scatter([np.mean(lab)], [y], marker="D", s=110, facecolor="none", edgecolor=INK2, lw=1.2, zorder=4)
        else:
            b = [cover[f"{m}/{s}"]["d8"] for s in seeds]
            a = [cover[f"{m}{sfx}/{s}"]["d8"] for s in seeds]
        a2.scatter(b, [y] * len(b), s=14, facecolor="none", edgecolor=col, lw=0.8, alpha=0.7, zorder=3)
        a2.scatter(a, [y] * len(a), s=14, color=col, lw=0, alpha=0.7, zorder=3)
        if abs(np.mean(a) - np.mean(b)) > 3:
            a2.annotate("", xy=(np.mean(a), y), xytext=(np.mean(b), y), zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, shrinkA=6, shrinkB=6))
        a2.scatter([np.mean(b)], [y], s=50, facecolor=SURF, edgecolor=col, lw=1.6, zorder=5)
        a2.scatter([np.mean(a)], [y], s=50, color=col, edgecolor=SURF, lw=1.0, zorder=5)
    nn_x = [cover[f"{m}{sfx}/{s}"].get("nn_extra", cover[f"{m}{sfx}/{s}"].get("nn_dagger")) for m, _, _ in models for s in seeds]
    nn_ok = [v["ok"]["nn"] for v in dag.values() if "ok" in v]
    a2.text(0.5, 2.02, f"测试时卡住的状态（没见过：离 {what} 数据 {min(nn_x):.2f}–{max(nn_x):.2f}"
            + (f"，正常状态离示范约 {np.mean(nn_ok):.2f}）" if nn_ok else "）"), fontsize=8.5, color=INK)
    a2.set_yticks(list(rows.values()), ["NCA" if m == "se_global" else "小 DP" for _, m in rows], fontsize=8.5)
    a2.set_ylim(0.1, heads[0] + 0.33)
    far = [fit[f"{m}/{s}"][g][x]["d8"] for g, _ in groups for m, _, _ in models for s in seeds for x in ("before", "after")]
    a2.set_xlim(0, max(70, 10 * np.ceil((max(far) + 4) / 10)))         # DART 的远状态上计划走得更远（约 80 像素）
    a2.tick_params(axis="y", length=0)
    a2.grid(axis="x", color=GRID, lw=0.8)
    a2.set_xlabel("计划第 8 步离当前推杆位置（像素，中位数；示范状态上约 66）")
    a2.legend(handles=[Line2D([], [], ls="none", marker="o", ms=7, mfc=SURF, mec=INK2, mew=1.4, label=f"{what} 前"),
                       Line2D([], [], ls="none", marker="o", ms=7, mfc=INK2, mec=INK2, label=f"{what} 后"),
                       Line2D([], [], ls="none", marker="D", ms=8, mfc="none", mec=INK2, mew=1.2, label="大 DP 的标签")],
              loc="lower right", fontsize=8, frameon=False)
    a2.set_title(cfg["right_title"], fontsize=9.5, loc="left")
    fig.suptitle(cfg["title"], x=0.06, y=0.995, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.06, 1 - 0.37 / H, cfg["note"] + "\n彻底失败率和测试时卡住的状态来自每个种子 500 个测试回合（测试种子 100000 起）。"
             "细线、小点 = 种子（0–2），粗线、大点 = 均值。", fontsize=8, color=INK2, va="top", linespacing=1.5)
    fig.savefig(cfg["out"], dpi=150, bbox_inches="tight")
    print("saved", cfg["out"])


# 第九轮（按结果挑标签）的图 rescue.png，第十轮（再收集一轮）的图 rescue2.png：同样的四块，配置不同
RESCUE_CTRL_LAB = [("dp", "老师（大 DP）"), ("dp_noise", "老师 + 扰动（σ = 15 像素）"), ("dp_small", "小 DP（只用示范）"),
                   ("nca", "NCA（只用示范）")]
FIT_GROUPS = [("seen", "救出来的卡住状态（训练时见过，标签 = 救出来的那段动作）"),
              ("unrescued", "接手都没救出来的卡住状态（训练集里没有）"),
              ("test", "测试时 DART 后的模型卡住的状态"),
              ("own", "测试时补数据后的模型自己卡住的状态")]
RESCUE_FIGS = {
    "rescue": dict(
        out=os.path.join(HERE, "results", "pusht_tune", "rescue.png"),
        rounds=[("", "只用示范"), ("_dart", "+ DART"), ("_rescue", "+ 按结果挑标签")],          # 后缀 → 横轴
        students=[("se_global_dart", "NCA + DART", COL["nca"]), ("dp_small_dart", "小 DP + DART", COL["dp_small"])],
        ctrl_lab=RESCUE_CTRL_LAB + [("any", "12 次里任一次")],
        fit_pairs=[("se_global_rescue", "NCA", COL["nca"]), ("dp_small_rescue", "小 DP", COL["dp_small"])],
        fit_groups=FIT_GROUPS, fit_legend=("补数据前（+ DART）", "补数据后（+ 按结果挑标签）"),
        take_src=[("se_global_dart", "NCA + DART"), ("se_global_rescue", "NCA + 按结果挑标签"),
                  ("dp_small_dart", "小 DP + DART"), ("dp_small_rescue", "小 DP + 按结果挑标签")],
        title="按结果挑标签：NCA 的彻底失败降到 3.4%，得分 0.963、成功率 77.5%，超过同样补了数据的小 DP 和老师（大 DP）",
        rounds_title="彻底失败率（括号里是 3 个种子的平均得分）",
        data_title="谁能把学生救出来：只用示范的小 DP、NCA 最多，老师最少",
        fit_title="计划第 8 步：见过的卡住状态上 NCA 从 7 像素到 34（标签 47），新卡住的地方仍是 7–9",
        take_title="接管检验：NCA 卡住的回合 103 → 41 个（都是 1500 个测试回合里），剩下的老师能救出 29%（之前 10%），小 DP 44%",
        note="学生 = DART 后的模型。它们卡住时，老师、老师 + 扰动、只用示范的小 DP 和 NCA 各接手 3 次，只留救出来（最好覆盖率 ≥ 0.5）"
             "的那段实际执行的动作当标签，\n和示范、DART 数据一起从头训练（NCA 7.1–7.7 万轮，小 DP 10.5–11.4 万轮）。"),
    "rescue2": dict(
        out=os.path.join(HERE, "results", "pusht_tune", "rescue2.png"),
        rounds=[("", "只用示范"), ("_dart", "+ DART"), ("_rescue", "+ 按结果\n挑标签"), ("_rescue2", "+ 再收集\n一轮")],
        students=[("se_global_rescue", "NCA", COL["nca"]), ("dp_small_rescue", "小 DP", COL["dp_small"])],   # 图例短一点，不压住老师那行
        ctrl_lab=RESCUE_CTRL_LAB + [("dp_small_r9", "第九轮的小 DP"), ("nca_r9", "第九轮的 NCA"),
                                    ("base", "原来四个里任一次"), ("any", "18 次里任一次")],
        fit_pairs=[("se_global_rescue2", "NCA", COL["nca"]), ("dp_small_rescue2", "小 DP", COL["dp_small"])],
        fit_groups=FIT_GROUPS[:2] + [("test", "测试时第九轮的模型卡住的状态"), ("own", "测试时第十轮的模型自己卡住的状态")],
        fit_legend=("补数据前（第九轮）", "补数据后（第十轮）"),
        take_src=[("se_global_rescue", "NCA 第九轮"), ("se_global_rescue2", "NCA 第十轮"),
                  ("dp_small_rescue", "小 DP 第九轮"), ("dp_small_rescue2", "小 DP 第十轮")],
        title="再收集一轮：彻底失败 NCA 3.4% → 2.7%，小 DP 3.9% → 1.9%；得分持平（0.970 对 0.973），NCA 成功率仍高 11 个百分点",
        rounds_title="彻底失败率（括号里是 3 个种子的平均得分）",
        data_title="谁能把第九轮的模型救出来：NCA 卡住的救出 70%，\n小 DP 卡住的 52%；NCA 换随机数接手自己只救出 14%",
        fit_title="计划第 8 步：见过的卡住状态上 NCA 从 7–8 像素到 22–27（标签 29–34），新卡住的地方仍是 5–7",
        take_title="接管检验：卡住的回合 NCA 41 → 32 个，小 DP 36 → 14 个（都是 1500 个测试回合里）；剩下的老师和小 DP 只能救出 21–29%",
        note="学生 = 第九轮的模型。它们卡住时，第九轮的四个接手者和第九轮的两个模型各接手 3 次（第九轮的模型接手自己 = 换随机数），"
             "只留救出来的那段动作当标签，\n和示范、DART、第九轮的数据一起从头训练（NCA 9.4–10.3 万轮，小 DP 12.4–13.5 万轮）。"),
}


def panel_rounds(ax, final, title, rounds):
    """彻底失败率随补数据的轮次（final.json，每次新 z / 新噪声）：细线 = 种子，粗线 = 均值，点旁边是得分（3 个种子的均值）。
    每个轮次上，两条均值线里高的那个标在点上方，低的标在下方（第九轮 NCA 降到了小 DP 下面）。"""
    x = np.arange(len(rounds))
    series = []
    for m, lab, col in (("se_global", "NCA", COL["nca"]), ("dp_small", "小 DP", COL["dp_small"])):
        names = [m + sfx for sfx, _ in rounds]
        if not all(f"{n}/fresh/{s}" in final for n in names for s in "012"):
            continue
        cat = np.array([[final[f"{n}/fresh/{s}"]["catastrophic"] * 100 for s in "012"] for n in names])   # (轮次, 种子)
        score = [np.mean([final[f"{n}/fresh/{s}"]["score"] for s in "012"]) for n in names]
        for k in range(cat.shape[1]):
            ax.plot(x, cat[:, k], color=col, lw=0.9, alpha=0.35, zorder=2)
        ax.plot(x, cat.mean(1), color=col, lw=2.2, marker="o", ms=6, mec=SURF, mew=1, zorder=3, label=lab)
        series.append((cat.mean(1), score))
    for xi in x:
        hi = max(range(len(series)), key=lambda i: series[i][0][xi]) if series else 0
        for i, (c, sc) in enumerate(series):
            side = 1 if i == hi else -1
            ax.annotate(f"{c[xi]:.1f}%（{sc[xi]:.3f}）", (xi, c[xi]), xytext=(0, 8 * side), textcoords="offset points",
                        ha="center", va="bottom" if side > 0 else "top", fontsize=7.5, color=INK2, zorder=6,
                        bbox=dict(boxstyle="square,pad=0.1", fc=SURF, ec="none"))   # 底色盖住后面的虚线和种子细线
    dp = np.mean([final[f"dp/fresh/{s}"]["catastrophic"] * 100 for s in "012"])
    ax.axhline(dp, color=COL["dp"], lw=1.0, ls=(0, (4, 3)), zorder=0, label=f"老师（大 DP）{dp:.1f}%")
    ax.set_xticks(x, [lab for _, lab in rounds])
    ax.set_xlim(-0.5, len(rounds) - 0.5)                                 # 第一个点的标注不碰纵轴的刻度
    ax.set_ylim(0, 12)
    ax.set_ylabel("彻底失败率（%，500 回合）")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(title, fontsize=9.5, loc="left")


def panel_rescue_data(ax, rescue, title, students, ctrl_lab):
    """rescue.json：学生卡住的状态里，每个接手的规划器（3 次里任一次）救出的比例；any = 所有接手里任一次，
    base = 第九轮那四个接手者里任一次。小点 = 种子，大点 = 3 个种子合起来。"""
    ys = np.arange(len(ctrl_lab))[::-1].astype(float)
    for (c, lab), y in zip(ctrl_lab, ys):
        for (m, _, col), dy in zip(students, (0.16, -0.16)):
            vs = [rescue[f"{m}/{s}"] for s in "012" if f"{m}/{s}" in rescue and rescue[f"{m}/{s}"]["n_stall"]]
            if not vs:
                continue
            frac = {"any": lambda v: v["n_rescued"] / v["n_stall"],
                    "base": lambda v: v.get("base_any", v["n_rescued"] / v["n_stall"])}.get(c, lambda v: v["by_ctrl"][c]["any"])
            per = [100 * frac(v) for v in vs]
            pooled = 100 * sum(frac(v) * v["n_stall"] for v in vs) / sum(v["n_stall"] for v in vs)
            ax.scatter(per, [y + dy] * len(per), s=13, color=col, alpha=0.55, lw=0, zorder=3)
            ax.scatter([pooled], [y + dy], s=58, color=col, edgecolor=SURF, lw=1.0, zorder=4)
            ax.text(max(per + [pooled]) + 2.5, y + dy, f"{pooled:.0f}%", fontsize=7.5, color=INK2, va="center")   # 放在种子点右边
    ax.axhline(ys[-1] + 0.5, color=GRID, lw=0.8)
    ax.set_yticks(ys, [lab for _, lab in ctrl_lab], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(ys.min() - 0.5, ys.max() + 0.5)
    ax.grid(axis="x", color=GRID, lw=0.8)
    n = {m: sum(rescue[f"{m}/{s}"]["n_stall"] for s in "012" if f"{m}/{s}" in rescue) for m, _, _ in students}
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=7, mfc=col, mec=col, label=f"{lab} 卡住的状态（{n[m]} 个）")
                       for m, lab, col in students], loc="upper right", fontsize=8, frameon=False,
              handlelength=1.0, handletextpad=0.5)                       # 老师那行只到 30% 左右；图例收窄，不碰那行的标注
    ax.set_xlabel("救出（整回合最好覆盖率 ≥ 0.5）的比例（%）；每个规划器接手 3 次，任一次救出就算")
    ax.set_title(title, fontsize=9.5, loc="left")


def panel_rescue_fit(ax, fit, title, pairs, groups, legend):
    """rescue_fit.json：四组卡住的状态上，补数据前（空心圆）、后（实心圆）的计划第 8 步；橙色竖线 = 老师在同样状态上的计划，
    菱形 = 标签（救出来的那段动作）。小点 = 种子，大点 = 均值。"""
    seeds = "012"
    heads = [0.4 + 2.5 * (len(groups) - k) for k in range(len(groups))]
    ticks, labels, xmax = [], [], 70
    for (g, text), yh in zip(groups, heads):
        ax.text(0.5, yh, text, fontsize=8.5, color=INK)
        if g != groups[-1][0]:
            ax.axhline(yh - 2.12, color=GRID, lw=0.8)
        for (m, lab, col), y in zip(pairs, (yh - 0.65, yh - 1.45)):
            vs = [fit[f"{m}/{s}"][g] for s in seeds if f"{m}/{s}" in fit and g in fit[f"{m}/{s}"]]
            if not vs:
                continue
            b, a, t = ([v[x]["d8"] for v in vs] for x in ("before", "after", "teacher"))
            if "label8" in vs[0]:
                ax.scatter([np.mean([v["label8"] for v in vs])], [y], marker="D", s=95, facecolor="none", edgecolor=INK2,
                           lw=1.1, zorder=4)
            ax.scatter([np.mean(t)], [y], marker="|", s=170, color=COL["dp"], lw=2.0, zorder=4)
            ax.scatter(b, [y] * len(b), s=14, facecolor="none", edgecolor=col, lw=0.8, alpha=0.7, zorder=3)
            ax.scatter(a, [y] * len(a), s=14, color=col, lw=0, alpha=0.7, zorder=3)
            if abs(np.mean(a) - np.mean(b)) > 3:
                ax.annotate("", xy=(np.mean(a), y), xytext=(np.mean(b), y), zorder=4,
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, shrinkA=6, shrinkB=6))
            ax.scatter([np.mean(b)], [y], s=50, facecolor=SURF, edgecolor=col, lw=1.6, zorder=5)
            ax.scatter([np.mean(a)], [y], s=50, color=col, edgecolor=SURF, lw=1.0, zorder=5)
            ticks.append(y)
            labels.append(f"{lab}（{sum(v['n'] for v in vs)}）")
            xmax = max(xmax, *a, *b, *t, *([v["label8"] for v in vs] if "label8" in vs[0] else []))
    ax.set_yticks(ticks, labels, fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(heads[-1] - 2.0, heads[0] + 0.45)
    ax.set_xlim(0, 10 * np.ceil((xmax + 4) / 10))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("计划第 8 步离当前推杆位置（像素，中位数；示范状态上约 60）；括号里是状态数")
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=7, mfc=SURF, mec=INK2, mew=1.4, label=legend[0]),
                       Line2D([], [], ls="none", marker="o", ms=7, mfc=INK2, mec=INK2, label=legend[1]),
                       Line2D([], [], ls="none", marker="|", ms=11, mew=2.0, color=COL["dp"], label="老师（大 DP）"),
                       Line2D([], [], ls="none", marker="D", ms=7.5, mfc="none", mec=INK2, mew=1.1, label="标签")],
              loc="lower right", fontsize=8, frameon=False)
    ax.set_title(title, fontsize=9.5, loc="left")


def rescue_figure(final, rescue, fit, take, key="rescue"):
    """按结果挑标签（RESCUE_FIGS[key]）：左上彻底失败率随轮次，左中数据（谁能把学生救出来），右边计划第 8 步，下面接管检验。"""
    cfg = RESCUE_FIGS[key]
    H = 11.6 + 0.3 * (len(cfg["ctrl_lab"]) - 5)                          # 接手者多时左中那块高一点
    fig = plt.figure(figsize=(12.5, H))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.55], height_ratios=[3.7, 3.4 + 0.3 * (len(cfg["ctrl_lab"]) - 5), 2.6],
                          wspace=0.34, hspace=0.42)
    fig.subplots_adjust(top=1 - 1.25 / H)
    panel_rounds(fig.add_subplot(gs[0, 0]), final, cfg["rounds_title"], cfg["rounds"])
    panel_rescue_data(fig.add_subplot(gs[1, 0]), rescue, cfg["data_title"], cfg["students"], cfg["ctrl_lab"])
    panel_rescue_fit(fig.add_subplot(gs[0:2, 1]), fit, cfg["fit_title"], cfg["fit_pairs"], cfg["fit_groups"], cfg["fit_legend"])
    if take:
        panel_takeover(fig.add_subplot(gs[2, :]), take, cfg["take_title"], cfg["take_src"])
    fig.suptitle(cfg["title"], x=0.06, y=0.995, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.06, 1 - 0.37 / H, cfg["note"] + "\n彻底失败率、测试时卡住的状态和接管检验来自每个种子 500 个测试回合（测试种子 100000 起）。"
             "细线、小点 = 种子（0–2），粗线、大点 = 均值或 3 个种子合起来。", fontsize=8, color=INK2, va="top", linespacing=1.5)
    fig.savefig(cfg["out"], dpi=150, bbox_inches="tight")
    print("saved", cfg["out"])


# T 块的两个矩形（gym_pusht 的 add_tee，scale 30，length 4），块坐标系；和 viz_pusht.py 一样
T_SHAPES = [np.array([(-60, 30), (60, 30), (60, 0), (-60, 0)], float), np.array([(-15, 30), (-15, 120), (15, 120), (15, 30)], float)]
GOAL_POSE, GOAL = (256.0, 256.0, np.pi / 4), "#d9d8d3"
PATH_NAMES = {"se_global": "NCA（se_global）", "se_global_dag": "NCA + DAgger", "dp_small": "小 DP", "dp_small_dag": "小 DP + DAgger"}
PATHS_TITLE = "闭环路径：DAgger 后的模型卡在了另一批回合上，和换一个训练种子差不多"


def tee(ax, pose, **kw):
    x, y, th = pose
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for v in T_SHAPES:
        ax.add_patch(Polygon(v @ R.T + (x, y), closed=True, **kw))


def paths_figure(pj):
    """paths.json（pusht_tune.py --stage paths）→ paths.png：行 = 模型，列 = 回合。"""
    cols = [c for c in pj["cols"] if c["episode"] is not None]
    models = pj["models"]
    fig, axs = plt.subplots(len(models), len(cols), figsize=(2.75 * len(cols) + 0.6, 2.9 * len(models) + 1.6), squeeze=False,
                            gridspec_kw=dict(hspace=0.22, wspace=0.05))
    fig.subplots_adjust(top=0.825, left=0.07, right=0.99, bottom=0.01)
    for j, c in enumerate(cols):
        e = c["episode"]
        for i, m in enumerate(models):
            ax, r = axs[i, j], pj["traj"][f"{m}/{e}"]
            col = COL["dp_small"] if m.startswith("dp") else COL["nca"]
            ax.add_patch(Polygon([(0, 0), (512, 0), (512, 512), (0, 512)], closed=True, fc="white", ec=GRID, lw=0.8, zorder=0))
            tee(ax, GOAL_POSE, fc=GOAL, ec="none", zorder=1)
            tee(ax, r["block"][0], fill=False, ec=INK2, lw=0.9, ls=(0, (3, 2)), zorder=2)
            tee(ax, r["block_end"], fc=col, ec=col, alpha=0.35, lw=1.0, zorder=3)
            ag = np.array(r["agent"])
            ax.plot(ag[:, 0], ag[:, 1], color=col, lw=1.0, zorder=4)
            ax.plot(*ag[0], "o", ms=4.5, mfc="white", mec=INK, mew=1.0, zorder=6)
            st = r.get("stall")
            if st:                                                         # 第一次卡住：主图标位置，离它最远的角上放大
                t, p0, W = st["t"], ag[st["t"]], 90
                ax.plot(*p0, "o", ms=6, mfc="none", mec=INK, mew=1.2, zorder=7)
                ax.add_patch(Rectangle((p0[0] - W, p0[1] - W), 2 * W, 2 * W, fill=False, ec=INK2, lw=0.6, ls=(0, (2, 2)), zorder=6))
                fx, fy = p0[0] / 512, 1 - p0[1] / 512                      # 坐标轴分数（y 轴是反的）
                cx, cy = max([(0.57, 0.57), (0.01, 0.57), (0.01, 0.01), (0.57, 0.01)],
                             key=lambda c: (c[0] + 0.21 - fx) ** 2 + (c[1] + 0.21 - fy) ** 2)
                ins = ax.inset_axes([cx, cy, 0.42, 0.42], zorder=10)
                ins.set_facecolor("white")
                tee(ins, st["block"], fc=col, ec=col, alpha=0.25, lw=0.8, zorder=1)
                seg = ag[max(t - 16, 0):t + 25]                            # 卡住前 16 步到之后 24 步的推杆轨迹
                ins.plot(seg[:, 0], seg[:, 1], color=col, lw=1.2, zorder=2)
                for p in st["dp_plan"]:
                    q = np.vstack([p0, p])
                    ins.plot(q[:, 0], q[:, 1], color=COL["dp"], lw=1.0, ls=(0, (3, 1.5)), zorder=3)
                for p in st["plan"]:
                    q = np.vstack([p0, p])
                    ins.plot(q[:, 0], q[:, 1], color=INK, lw=0.8, marker="o", ms=2.0, zorder=4)
                ins.plot(*p0, "o", ms=6, mfc="none", mec=INK, mew=1.2, zorder=5)
                ins.set(xlim=(p0[0] - W, p0[0] + W), ylim=(p0[1] + W, p0[1] - W), aspect="equal", xticks=[], yticks=[])
                for sp in ins.spines.values():
                    sp.set_visible(True)
                    sp.set_edgecolor(INK2)
                    sp.set_linewidth(0.7)
            ax.set(xlim=(0, 512), ylim=(512, 0), aspect="equal", xticks=[], yticks=[])   # y 向下，和环境渲染的画面一致
            for sp in ax.spines.values():
                sp.set_visible(False)
            res = "成功" if r["success"] else ("彻底失败" if r["best"] < 0.5 else "没成功")
            ax.set_title(f"{res}：最好覆盖 {r['best']:.2f}，{r['steps']} 步", fontsize=8, loc="left", color=INK2, pad=3)
            if j == 0:
                ax.set_ylabel(PATH_NAMES.get(m, m), fontsize=9.5, color=INK, labelpad=6)
        axs[0, j].annotate(f"{c['label']}\n回合 {e}（测试种子 {100000 + e}），500 个里共 {c['n_match']} 个",
                           xy=(0, 1), xycoords="axes fraction", xytext=(0, 20), textcoords="offset points",
                           fontsize=8.5, color=INK, va="bottom", linespacing=1.5)
    ov = pj.get("overlap", {})

    def line(m, name):                                                    # 彻底失败的回合重合多少，DAgger 前后和两种对照
        g = lambda k: ov.get(f"{m}|{m}{k}", {})
        if not g("_dag"):
            return ""
        return (f"{name}：DAgger 前后彻底失败 {g('_dag')['a']} → {g('_dag')['b']} 个，重合 {g('_dag')['both']} 个；"
                f"对照：同一个模型换随机数重跑重合 {g('@rerun').get('both', '–')} 个（{g('@rerun').get('b', '–')} 个里），"
                f"换一个训练种子的模型重合 {g('#other').get('both', '–')} 个（{g('#other').get('b', '–')} 个里）")
    fig.suptitle(PATHS_TITLE, x=0.02, y=0.995, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.02, 0.968, f"种子 {pj['seed']}，最终对比的 {pj['episodes']} 个测试回合；每列按规则取第一个符合的回合，同一列的初始状态相同。\n"
             + line("se_global", "NCA") + "。\n" + line("dp_small", "小 DP") + "。",
             fontsize=8.5, color=INK2, va="top", linespacing=1.6)
    fig.legend(handles=[Patch(fc=GOAL, ec="none", label="目标位姿"),
                        Patch(fill=False, ec=INK2, ls=(0, (3, 2)), label="初始 T 块"),
                        Patch(fc=COL["nca"], alpha=0.35, ec=COL["nca"], label="结束时的 T 块（颜色 = 模型）"),
                        Line2D([], [], color=INK2, lw=1.0, label="推杆轨迹（○ = 起点）"),
                        Line2D([], [], ls="none", marker="o", ms=6, mfc="none", mec=INK, label="第一次卡住的位置"),
                        Patch(fill=False, ec=INK2, ls=(0, (2, 2)), label="角上的小图 = 卡住处放大（边长 180 像素）"),
                        Line2D([], [], color=INK, lw=0.8, marker="o", ms=2.0, label="那一刻模型重新采样的 3 个计划"),
                        Line2D([], [], color=COL["dp"], lw=1.0, ls=(0, (3, 1.5)), label="同一观测上大 DP 的 3 个计划")],
               loc="upper left", bbox_to_anchor=(0.02, 0.918), ncol=4, fontsize=8, frameon=False, columnspacing=1.4)
    fig.savefig(PATHS_OUT, dpi=150, bbox_inches="tight")
    print("saved", PATHS_OUT)


# 大样本对比（pusht_tune.py --stage big，NOTES.md 19.11）
BIG_OUT = os.path.join(HERE, "results", "pusht_tune", "big.png")
BIG_FAM = [("se_global", "NCA", COL["nca"]), ("dp_small", "小 DP", COL["dp_small"])]   # 第九轮 = <名字>_rescue，第十轮 = _rescue2
BIG_ROUNDS = [("_rescue", "第九轮"), ("_rescue2", "第十轮")]
BIG_METRICS = [("score", "得分", 1, 3), ("success", "成功率（%）", 100, 1), ("catastrophic", "彻底失败率（%）", 100, 1)]
BIG_CMP = [("se_global_rescue2|dp_small_rescue2", "NCA − 小 DP\n（第十轮）", INK),            # 配对比较（big.json 的 pairs）
           ("se_global_rescue|dp_small_rescue", "NCA − 小 DP\n（第九轮）", INK2),
           ("se_global_rescue2|se_global_rescue", "NCA\n第十轮 − 第九轮", COL["nca"]),
           ("dp_small_rescue2|dp_small_rescue", "小 DP\n第十轮 − 第九轮", COL["dp_small"])]
BIG_FIG = dict(
    title="大样本对比（每个种子 8000 个新回合）：第十轮 NCA 和小 DP 得分持平，NCA 成功率高 9 个点，彻底失败多 0.45 个点",
    slope_titles=["得分：第十轮持平（0.964 对 0.965）\n新回合上都比测试回合低一点",
                  "成功率：NCA 两轮都高 9–11 个点",
                  "彻底失败：第九轮持平（4.1% 对 4.0%），\n第十轮小 DP 少（2.9% 对 3.3%）"],
    forest_titles=["逐回合配对的得分差：第十轮 NCA − 小 DP\n−0.001 ± 0.001；两个模型都比第九轮高",
                   "成功率差：NCA 比小 DP 高 9–11 个点\n（3 个种子都高）",
                   "彻底失败率差：第十轮 NCA 多 0.45 个点\n（3 个种子都多）；小 DP 降得更多"],
)


def panel_big_slope(ax, big, final, key, ylab, k, nd, title, legend=False):
    """第九轮 → 第十轮：细线 = 种子，粗线、实心点 = 3 个种子的均值（大样本），空心点、虚线 = 500 个测试回合（final.json）。
    均值的数字标在点的外侧（第九轮在左，第十轮在右），不压住同一列的空心点；两个里高的往上挪一点，低的往下挪一点。"""
    x = np.arange(len(BIG_ROUNDS))
    means = []
    for fam, lab, col in BIG_FAM:
        names = [fam + sfx for sfx, _ in BIG_ROUNDS]
        if not all(f"{n}/{s}" in big["models"] for n in names for s in "012"):
            continue
        v = np.array([[big["models"][f"{n}/{s}"][key] * k for s in "012"] for n in names])   # (轮次, 种子)
        for j in range(v.shape[1]):
            ax.plot(x, v[:, j], color=col, lw=0.9, alpha=0.35, zorder=2)
        ax.plot(x, v.mean(1), color=col, lw=2.2, marker="o", ms=6, mec=SURF, mew=1, zorder=4, label=lab)
        if all(f"{n}/fresh/{s}" in final for n in names for s in "012"):
            t = [np.mean([final[f"{n}/fresh/{s}"][key] * k for s in "012"]) for n in names]
            ax.plot(x, t, color=col, lw=1.0, ls=(0, (3, 2)), marker="o", ms=5, mfc=SURF, mec=col, mew=1.2, zorder=3)
        means.append(v.mean(1))
    for xi in x:
        hi = max(range(len(means)), key=lambda i: means[i][xi]) if means else 0
        for i, mv in enumerate(means):
            side, out = (1 if i == hi else -1), (-1 if xi == 0 else 1)
            ax.annotate(f"{mv[xi]:.{nd}f}", (xi, mv[xi]), xytext=(9 * out, 5 * side), textcoords="offset points",
                        ha="right" if out < 0 else "left", va="center", fontsize=8, color=INK2, zorder=6,
                        bbox=dict(boxstyle="square,pad=0.1", fc=SURF, ec="none"))
    ax.set_xticks(x, [lab for _, lab in BIG_ROUNDS])
    ax.set_xlim(-0.55, len(x) - 0.45)
    lo, hi_ = ax.get_ylim()
    ax.set_ylim(lo - 0.12 * (hi_ - lo), hi_ + 0.12 * (hi_ - lo))           # 给上下的数字留地方
    ax.set_ylabel(ylab)
    ax.grid(axis="y", color=GRID, lw=0.8)
    if legend:
        h = [Line2D([], [], color=c, lw=2.2, marker="o", ms=6, mec=SURF, label=f"{lab}（大样本）") for _, lab, c in BIG_FAM]
        h.append(Line2D([], [], color=INK2, lw=1.0, ls=(0, (3, 2)), marker="o", ms=5, mfc=SURF, mec=INK2, label="500 个测试回合"))
        ax.legend(handles=h, loc="best", fontsize=7.5, frameon=False)
    ax.set_title(title, fontsize=9.5, loc="left")


def panel_big_forest(ax, big, key, k, nd, title, labels=False):
    """BIG_CMP 里每个比较：小点 = 种子，大点和横线 = 3 个种子合起来的差和 95% 区间（只算回合的随机性）。
    得分差的标准误来自逐回合差；成功率、彻底失败率差的标准误来自配对的不一致回合数（b、c）：√(b + c − (b − c)² / n) / n。"""
    ys = np.arange(len(BIG_CMP))[::-1]
    span = 0.0
    for y, (pk, lab, col) in zip(ys, BIG_CMP):
        allp = big["pairs"].get(f"{pk}/all")
        if allp is None:
            continue
        per = [big["pairs"][f"{pk}/{s}"] for s in "012" if f"{pk}/{s}" in big["pairs"]]
        if key == "score":
            d = [p["d_score"] * k for p in per]
            m, se = allp["d_score"] * k, allp["se_score"] * k
            z = allp["d_score"] / allp["se_score"]
        else:
            q, n = allp[key], allp["n"]
            d = [p[key]["diff"] * k for p in per]
            m = q["diff"] * k
            se = np.sqrt(max(q["only_a"] + q["only_b"] - (q["only_a"] - q["only_b"]) ** 2 / n, 0.0)) / n * k
            z = q["z"]
        ax.scatter(d, [y + 0.2] * len(d), s=14, color=col, alpha=0.45, lw=0, zorder=3)
        ax.errorbar([m], [y], xerr=[[1.96 * se], [1.96 * se]], fmt="o", ms=6.5, color=col, mec=SURF, mew=1, elinewidth=1.8,
                    capsize=0, zorder=4)
        ax.annotate(f"{m:+.{nd}f}（z = {z:+.1f}）", (m, y), xytext=(0, -9), textcoords="offset points", ha="center", va="top",
                    fontsize=7.5, color=INK2, zorder=6, bbox=dict(boxstyle="square,pad=0.1", fc=SURF, ec="none"))
        span = max(span, abs(m) + 1.96 * se, *[abs(v) for v in d])
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_xlim(-1.3 * span, 1.3 * span)
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 0.55)
    ax.set_yticks(ys, [lab for _, lab, _ in BIG_CMP] if labels else [], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("前者 − 后者" + ("" if key == "score" else "（百分点）"))
    ax.set_title(title, fontsize=9.5, loc="left")


def big_figure(big, final):
    """大样本对比：上面一行是三个指标随轮次的变化（大样本和 500 个测试回合），下面一行是同样回合上的配对差。"""
    cfg = BIG_FIG
    H = 9.0
    fig, axs = plt.subplots(2, 3, figsize=(12.5, H), gridspec_kw=dict(height_ratios=[1, 1.1], hspace=0.36, wspace=0.26))
    fig.subplots_adjust(top=1 - 1.05 / H)
    for j, (key, ylab, k, nd) in enumerate(BIG_METRICS):
        panel_big_slope(axs[0, j], big, final, key, ylab, k, nd, cfg["slope_titles"][j], legend=j == 0)
        panel_big_forest(axs[1, j], big, key, k, nd if key == "score" else 2, cfg["forest_titles"][j], labels=j == 0)
    eps = sorted({v["episodes"] for kk, v in big["models"].items() if not kk.endswith("/all")})
    fig.suptitle(cfg["title"], x=0.06, y=0.995, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.06, 1 - 0.36 / H, f"每个种子 {'/'.join(map(str, eps))} 个新回合（种子 s 的环境种子从 600000 + 10000·s 起，和测试、"
             "补数据用过的都不重叠），都用每次新 z / 新噪声；同一个种子的模型跑同样的回合。彻底失败 = 最好覆盖率 < 0.5。\n"
             "上：细线 = 种子（0–2），粗线 = 3 个种子的均值；空心点、虚线 = 原来每个种子 500 个测试回合的均值。"
             "下：小点 = 种子，大点和横线 = 3 个种子合起来的差和 95% 区间（只算了回合的随机性，没算训练种子之间的差别）。",
             fontsize=8, color=INK2, va="top", linespacing=1.5)
    fig.savefig(BIG_OUT, dpi=150, bbox_inches="tight")
    print("saved", BIG_OUT)


def main():
    cat, att, steps, diag = load("cat.json"), load("attract.json"), load("steps.json"), load("diag.json")
    final = load("final.json") if os.path.exists(os.path.join(SRC, "final.json")) else {}
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 10.8), gridspec_kw=dict(hspace=0.38, wspace=0.28))   # 左上 28 行，高一点免得标签挤在一起
    panel_cat(axs[0, 0], cat, diag, final)
    panel_profile(axs[0, 1], att)
    panel_attract(axs[1, 0], att)
    ref = [float(np.mean(np.array(r["best"]) < 0.5)) * 100 for r in diag if r["model"] == "dp_small"]
    panel_steps(axs[1, 1], steps, float(np.mean(ref)))
    fig.suptitle("Push-T 上 NCA 的调参：只用示范时得分落后小 DP，差距来自分布外状态上的“慢启动”吸引子；按结果挑标签两轮后和同样处理的小 DP 持平",
                 x=0.06, y=0.995, ha="left",
                 fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.06, 0.958, "每次新 z 冷启动；每个（模型, 种子）100 个测试回合（测试种子 100000 起）；彻底失败 = 回合最好覆盖率 < 0.5。"
             "卡住的状态 = 彻底失败回合里推杆 16 步内移动不到 10 像素的时刻。线 = 所有（模型, 种子）的中位数，色带 = 最小到最大。",
             fontsize=8, color=INK2, wrap=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("saved", OUT)
    dag = load("dagger.json") if os.path.exists(os.path.join(SRC, "dagger.json")) else {}
    take = load("takeover.json") if os.path.exists(os.path.join(SRC, "takeover.json")) else None
    for key, name in (("dag", "dagger_fit.json"), ("dart", "dart_fit.json")):
        if os.path.exists(os.path.join(SRC, name)):
            dj = load(name)
            if "fit" in dj and "cover" in dj:
                dagger_figure(final, dj["fit"], dj["cover"], dag, key, take if key == "dart" else None)
    if all(os.path.exists(os.path.join(SRC, n)) for n in ("rescue.json", "rescue_fit.json")):
        rescue, fit = load("rescue.json"), load("rescue_fit.json")
        for key, cfg in RESCUE_FIGS.items():                             # 这一轮的模型训完、评估完才画
            if all(f"{m}/{s}" in fit for m, _, _ in cfg["fit_pairs"] for s in "012"):
                rescue_figure(final, rescue, fit, take, key)
    if os.path.exists(os.path.join(SRC, "paths.json")):
        paths_figure(load("paths.json"))
    if os.path.exists(os.path.join(SRC, "big.json")):
        big_figure(load("big.json"), final)


if __name__ == "__main__":
    main()
