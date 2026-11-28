#!/usr/bin/env python3
"""
轨迹动画：Push-T（状态版）上的闭环执行，做成 GIF（NOTES.md 第 19 节）。

traj.gif   第十轮的 NCA（se_global_rescue2）、同样补了数据的小 DP（dp_small_rescue2）和大 DP（老师），种子 0 的模型，
           在头两个测试回合（测试种子 100000、100001，没有挑选）上的闭环执行。
stall.gif  只用示范的 NCA（se_global）卡住的回合：同一批测试回合里它第一个彻底失败（最好覆盖率 < 0.5）的回合，
           以及第十轮的 NCA 在同一个回合上的执行（挑回合时不看第十轮的结果）。
都是每次规划新 z / 新噪声（和 final 一样），每 8 步重新规划。每个模型的 --n 个测试回合一批同步执行，随机数和 final
（500 个一批）不同，所以单个回合的结果不一定和 final 里一样。
画面：浅灰 = 目标位姿，彩色 = 当前的 T 块，黑圆 = 推杆，彩色线 = 推杆走过的路，点 = 当前计划的 16 步
（实心 = 这一块要执行的前 8 步，空心 = 后 8 步，执行不到，下一次规划会重新生成）。读数里的“卡住” = 过去 16 步推杆
移动不到 10 像素（只是显示用，和 stall.md 里“之后 8、16 步”的定义方向相反）。读数里的覆盖率 = 覆盖率奖励
clip(覆盖率 / 0.95, 0, 1)（和得分一样），向下取两位：显示 1.00 的只有成功的回合，0.996 显示成 0.99。
要 runs/ 里的模型和 gym_pusht（只在服务器上）。

用法：python viz_gif.py   →   results/pusht_tune/traj.gif、stall.gif（--tag 给文件名加后缀）
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon
from PIL import Image

import pusht as P
import pusht_tune as T

HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"nca": "#2a78d6", "dp": "#eb6834", "dp_small": "#1baf7a"}   # 和 viz_pusht.py、viz_tune.py 一致
INK, INK2, GRID, SURF, GOAL = "#1f1f1e", "#5f5e5a", "#e6e5e0", "#fcfcfb", "#d9d8d3"
NAME = {"se_global": "NCA（只用示范）", "se_global_dart": "NCA（DART 后）", "se_global_rescue": "NCA（第九轮）",
        "se_global_rescue2": "NCA（第十轮）", "dp_small": "小 DP（只用示范）", "dp_small_rescue": "小 DP（第九轮）",
        "dp_small_rescue2": "小 DP（第十轮）", "dp": "大 DP（老师）"}
NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"   # 服务器上的中文字体
fonts = ["Microsoft YaHei", "PingFang SC"]
if os.path.exists(NOTO):
    font_manager.fontManager.addfont(NOTO)
    fonts.insert(0, font_manager.FontProperties(fname=NOTO).get_name())
plt.rcParams.update({"font.sans-serif": fonts + ["DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.titlecolor": INK})

# T 块的两个矩形（gym_pusht 的 add_tee，scale = 30，length = 4），块坐标系；和 viz_pusht.py 相同
S, L = 30, 4
T_SHAPES = [np.array([(-L * S / 2, S), (L * S / 2, S), (L * S / 2, 0), (-L * S / 2, 0)], float),
            np.array([(-S / 2, S), (-S / 2, L * S), (S / 2, L * S), (S / 2, S)], float)]
GOAL_POSE = (256.0, 256.0, np.pi / 4)
AGENT_R = 15                                                         # 推杆半径（像素，gym_pusht）
STALL_WIN, STALL_PX = 16, 10.0                                       # 读数里的“卡住”：过去 16 步推杆移动不到 10 像素


def color(m):
    return COL["dp_small"] if m.startswith("dp_small") else COL["dp"] if m == "dp" else COL["nca"]


def cov(x):
    """覆盖率奖励向下取两位：四舍五入会把没成功的 0.996 显示成 1.00。"""
    return f"{np.floor(x * 100 + 1e-9) / 100:.2f}"


def tee(ax, pose, **kw):
    x, y, th = pose
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for v in T_SHAPES:
        ax.add_patch(Polygon(v @ R.T + (x, y), closed=True, **kw))


def record(m, n, seed, rng_seed):
    """模型 m（种子 seed）每次新 z / 新噪声闭环 n 个测试回合（测试种子 100000 起，一批同步执行，和 pusht.rollout 一样）。
    每个回合记下每一步之后的推杆位置、T 块位姿、覆盖率奖励，以及每次规划的 16 步计划（像素）。"""
    import gym_pusht  # noqa: F401  注册环境
    import gymnasium as gym
    torch.manual_seed(rng_seed)
    pl = P.make_planner(T.model_kind(m), T.model_path(m, seed), "fresh")
    envs = [gym.make("gym_pusht/PushT-v0", obs_type="state").unwrapped for _ in range(n)]
    states = np.stack([env.reset(seed=P.TEST_SEED + i)[0] for i, env in enumerate(envs)]).astype(np.float32)
    prev = states.copy()
    pl.reset(n)
    alive, success = np.ones(n, bool), np.zeros(n, bool)
    agent, block = [[s[:2].copy()] for s in states], [[s[2:5].copy()] for s in states]
    rew, plans = [[0.0] for _ in range(n)], [[] for _ in range(n)]       # 第 0 步还没有奖励，记 0
    for t0 in range(0, P.MAX_STEPS, P.T_A):
        obs = torch.from_numpy(np.concatenate([P.feat(states), P.feat(prev)], 1))
        plan = (pl.plan(obs).numpy() + 1.0) * 256.0                      # (n, 16, 2) 像素
        for e in np.flatnonzero(alive):
            plans[e].append(plan[e].copy())
        for k in range(min(P.T_A, P.MAX_STEPS - t0)):
            prev = states.copy()
            for e in np.flatnonzero(alive):
                s, r, term, _, _ = envs[e].step(plan[e, k])
                states[e] = s
                agent[e].append(s[:2].copy())
                block[e].append(s[2:5].copy())
                rew[e].append(float(r))
                if term:
                    success[e], alive[e] = True, False
            if not alive.any():
                break
        if not alive.any():
            break
    return [dict(agent=np.array(agent[e]), block=np.array(block[e]), rew=np.array(rew[e]), plans=np.array(plans[e]),
                 success=bool(success[e]), best=float(np.max(rew[e]))) for e in range(n)]


def draw(ax, ep, t, m):
    """回合记录 ep 执行了 t 个动作之后的画面（t 超过回合长度时停在最后一帧）。"""
    last = len(ep["agent"]) - 1
    t = min(t, last)
    ax.add_patch(Polygon([(0, 0), (512, 0), (512, 512), (0, 512)], closed=True, fc="white", ec=GRID, lw=0.8, zorder=0))
    tee(ax, GOAL_POSE, fc=GOAL, ec="none", zorder=1)
    tee(ax, ep["block"][t], fc=color(m), ec=color(m), alpha=0.45, lw=1.0, zorder=2)
    ag = ep["agent"][: t + 1]
    ax.plot(ag[:, 0], ag[:, 1], color=color(m), lw=1.3, alpha=0.85, solid_capstyle="round", zorder=3)
    if t < last:                                                         # 回合结束后不再有计划
        p = ep["plans"][min(t // P.T_A, len(ep["plans"]) - 1)]
        ax.plot(p[:, 0], p[:, 1], color=INK, lw=0.6, alpha=0.35, zorder=4)
        ax.scatter(p[:P.T_A, 0], p[:P.T_A, 1], s=11, color=INK, lw=0, zorder=5)
        ax.scatter(p[P.T_A:, 0], p[P.T_A:, 1], s=11, facecolor="white", edgecolor=INK, lw=0.7, zorder=5)
    ax.add_patch(Circle(ep["agent"][t], AGENT_R, fc=INK, ec="white", lw=1.0, zorder=6))
    ax.set(xlim=(-4, 516), ylim=(516, -4), aspect="equal", xticks=[], yticks=[])   # y 向下，和环境渲染的画面一致
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(NAME.get(m, m), fontsize=9, loc="left", pad=3)
    if t == last and ep["success"]:
        msg, col = f"第 {t} 步成功（覆盖率 {cov(ep['rew'][t])}）", INK
    elif t == last:
        msg = f"{t} 步用完，最好覆盖率 {cov(ep['best'])}" + ("，彻底失败" if ep["best"] < 0.5 else "")
        col = INK
    else:
        stuck = t >= STALL_WIN and np.linalg.norm(ep["agent"][t] - ep["agent"][t - STALL_WIN]) < STALL_PX
        msg = f"第 {t} 步　覆盖率 {cov(ep['rew'][t])}　最好 {cov(np.max(ep['rew'][: t + 1]))}" + ("　卡住" if stuck else "")
        col = INK if stuck else INK2
    ax.text(0.0, -0.025, msg, transform=ax.transAxes, fontsize=8, color=col, va="top", ha="left")


def render(cells, fig, every):
    """cells = [(ax, 回合记录, 模型)]：逐帧画到所有回合都结束，返回 PIL 图像列表。"""
    t_max = max(len(ep["agent"]) - 1 for _, ep, _ in cells)
    ts = list(range(0, t_max + 1, every))
    if ts[-1] != t_max:
        ts.append(t_max)
    frames = []
    for t in ts:
        for ax, ep, m in cells:
            ax.clear()
            draw(ax, ep, t, m)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    return frames


def save_gif(frames, path, ms=70, hold_ms=2500):
    """共用一个调色板（从首、中、尾三帧取），避免逐帧换色闪烁；最后一帧停 hold_ms。"""
    pick = [frames[i] for i in sorted({0, len(frames) // 2, len(frames) - 1})]
    w, h = frames[0].size
    sheet = Image.new("RGB", (w, h * len(pick)))
    for i, f in enumerate(pick):
        sheet.paste(f, (0, h * i))
    mediancut = getattr(getattr(Image, "Quantize", Image), "MEDIANCUT")   # 新旧版本的 Pillow 都能用
    no_dither = getattr(getattr(Image, "Dither", Image), "NONE")
    pal = sheet.quantize(colors=96, method=mediancut)
    q = [f.quantize(palette=pal, dither=no_dither) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=[ms] * (len(q) - 1) + [hold_ms], loop=0)
    print(f"{path}: {len(q)} 帧，{w}×{h}，{os.path.getsize(path) / 1e6:.1f} MB", flush=True)


PANEL, GAP, TOP, BELOW, LEFT, RIGHT, FOOT = 2.5, 0.12, 0.24, 0.24, 0.3, 0.1, 0.76   # 英寸
DPI = 110


def grid(R, C, head):
    """R × C 个正方形棋盘，按英寸排版：上面 head 英寸放总标题，每个棋盘上方放模型名、下方放读数，最下面四行图例。"""
    W = LEFT + C * PANEL + (C - 1) * GAP + RIGHT
    H = head + R * (TOP + PANEL + BELOW) + FOOT
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    axes = np.array([[fig.add_axes([(LEFT + c * (PANEL + GAP)) / W, (FOOT + (R - 1 - r) * (TOP + PANEL + BELOW) + BELOW) / H,
                                    PANEL / W, PANEL / H]) for c in range(C)] for r in range(R)])
    fig.text(0.1 / W, (FOOT - 0.1) / H, "浅灰 = 目标位姿　彩色块 = T 块　黑圆 = 推杆　彩色线 = 推杆走过的路\n"
             "实心点 = 这一块要执行的 8 步　空心点 = 计划的后 8 步（执行不到）\n"
             f"卡住 = 过去 {STALL_WIN} 步推杆移动不到 {STALL_PX:g} 像素\n"
             "覆盖率 = 实际覆盖率 / 0.95（封顶 1）；到 1.00 就算成功，回合结束", fontsize=7.5, color=INK2, ha="left",
             va="top", linespacing=1.5)
    return fig, axes, W, H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["se_global_rescue2", "dp_small_rescue2", "dp"], help="traj.gif 的列")
    ap.add_argument("--stall_models", nargs="+", default=["se_global", "se_global_rescue2"],
                    help="stall.gif 的列；第一个模型第一个彻底失败的回合")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1], help="traj.gif 的行：第几个测试回合（从 0 数）")
    ap.add_argument("--n", type=int, default=100, help="每个模型一批跑几个测试回合")
    ap.add_argument("--seed", type=int, default=0, help="模型的训练种子")
    ap.add_argument("--every", type=int, default=2, help="每隔几步画一帧")
    ap.add_argument("--tag", default="", help="输出文件名后缀")
    a = ap.parse_args()
    torch.set_num_threads(4)
    out = os.path.join(HERE, "results", "pusht_tune")
    os.makedirs(out, exist_ok=True)
    rec = {}
    for m in dict.fromkeys(a.models + a.stall_models):
        rec[m] = record(m, a.n, a.seed, a.seed + 1)
        eps = rec[m]
        print(f"{m}: {a.n} 个回合，得分 {np.mean([e['best'] for e in eps]):.3f}，成功 {np.mean([e['success'] for e in eps]):.2f}，"
              f"彻底失败 {np.mean([e['best'] < 0.5 for e in eps]):.2f}", flush=True)

    # traj.gif：行 = 测试回合，列 = 模型
    R, C = len(a.episodes), len(a.models)
    fig, axes, W, H = grid(R, C, 0.42)
    fig.text(0.1 / W, 1 - 0.12 / H, f"同一个测试回合上的 {C} 个模型（训练种子 {a.seed}，每次规划新 z / 新噪声，每 8 步重新规划）",
             fontsize=9.5, ha="left", va="top", color=INK)
    for r, e in enumerate(a.episodes):
        pos = axes[r, 0].get_position()
        fig.text(0.15 / W, (pos.y0 + pos.y1) / 2, f"测试回合 {P.TEST_SEED + e}", rotation=90, va="center", ha="center",
                 fontsize=8, color=INK2)
    cells = [(axes[r, c], rec[m][e], m) for r, e in enumerate(a.episodes) for c, m in enumerate(a.models)]
    save_gif(render(cells, fig, a.every), os.path.join(out, f"traj{a.tag}.gif"))
    plt.close(fig)

    # stall.gif：第一个模型第一个彻底失败的回合
    first = a.stall_models[0]
    bad = [i for i, ep in enumerate(rec[first]) if ep["best"] < 0.5]
    if not bad:
        print(f"{first} 在 {a.n} 个回合里没有彻底失败，不画 stall.gif")
        return
    e = bad[0]
    print(f"stall：测试回合 {P.TEST_SEED + e}（{first} 的第 1 个彻底失败，共 {len(bad)} 个）；"
          + "，".join(f"{m} 最好 {rec[m][e]['best']:.2f}" for m in a.stall_models), flush=True)
    C = len(a.stall_models)
    fig, axes, W, H = grid(1, C, 0.62)
    fig.text(0.1 / W, 1 - 0.1 / H, f"测试回合 {P.TEST_SEED + e}：前 {a.n} 个回合里 {NAME.get(first, first)}第一个彻底失败的\n"
             "卡住时计划的 16 步缩在推杆附近（慢启动吸引子）", fontsize=9, ha="left", va="top", color=INK, linespacing=1.5)
    save_gif(render([(axes[0, c], rec[m][e], m) for c, m in enumerate(a.stall_models)], fig, a.every),
             os.path.join(out, f"stall{a.tag}.gif"))
    plt.close(fig)


if __name__ == "__main__":
    main()
