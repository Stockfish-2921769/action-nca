# Action NCA：用神经元胞自动机“长出”机器人动作序列

这个项目是从 claude.ai 的一次对话迁移过来的。对话里的背景、设计理由、实验结果和被修正过的判断都整理在 NOTES.md，下面这行会把它一起载入：

@NOTES.md

## 目标

探索“不从噪声出发，用细胞涌现动作序列”：用一维神经元胞自动机（NCA）代替 Diffusion Policy 的去噪过程来生成动作块，并检验：

1. 只在种子细胞里放一个低维隐变量 z，能否表达多模态动作分布；
2. 把滚动时域执行做成“再生”（切掉已执行的细胞、前移、补尾、续长），是否比每次从头生成更省步数、模态更连贯；
3. 和 Diffusion Policy 相比孰优孰劣（尚未验证，是下一步的重点）。

## 文件

- `action_nca.py`：单文件原型，包括数据生成、模型、训练、闭环评估和画图。
- `sweep_steps.py`：加载训练好的隐变量模型，扫描冷启动和再生的 NCA 步数。
- `diffusion_baseline.py`：同一任务上的 Diffusion Policy 基线（条件 1D U-Net + DDIM，或 `--method flow` 流匹配 + Euler），三种执行方式（每次新噪声、固定噪声、SDEdit 式热启动），并用同一套 `rollout()` 评估 `nca_out/` 里的 NCA 模型做对比。输出在 `dp_out/`（已加入 .gitignore）。
- `results/`：上一次默认参数运行的 `metrics.json`、`results.png`、`sweep.txt`、`train_log.txt`，作为基线结果保留。
- `results/nca_rerun_local/`：本机（torch 2.12）按默认参数重训的 NCA 结果；`results/dp_baseline/`：DP 各配置的对比表、指标、图和日志。
- `precision_loss.py`：精度损失实验（`action_nca.py --prec_weight`），成对比较不同权重，汇总复制到 `results/precision_loss/`。
- `guidance.py`：推理时可微避障引导（interleave 边长边推、post 长完再推），加载 `runs/precision/` 里的模型评估，汇总复制到 `results/guidance/`。
- `novel_obstacle.py`：测试时新障碍（训练时没见过的圆），NCA 和小 DP 在同样的引导下比较（`--proj normal` 只推路径法向分量，结果在 `results/normal_push/`）；DP 模型在 `runs/novel/dp_seed<s>/`，汇总复制到 `results/novel_obstacle/`。
- `viz_regrow.py`：画第 14 节的图（再生 / 热启动 + interleave 的路径、最大拐折分布、逐种子成功率和拐折增量）。
- `results/push_training/`：推动训练后 NCA 的指标和新障碍评估（NOTES.md 第 15 节）。
- `viz_best.py`：画第 16 节最好的四种组合（路径、逐种子成功率和最大拐折、big 场景里只推法向的作用）。
- `random_obstacle.py`：障碍随机化、策略看见障碍（观测 = 位置 + 4 个障碍参数）；NCA / DP × 看见 / 看不见，in / ood 两组评估障碍。结果在 `results/random_obstacle/`（NOTES.md 第 17 节）。
- `multi_seed.py`：多种子实验（种子 0–4 的 NCA、小 DP、大 DP），调度训练和评估、汇总均值和波动、逐种子检验结论；中间结果在 `runs/`（已加入 .gitignore），汇总复制到 `results/multi_seed/`。
- `README.md`：对外的项目说明（GitHub 仓库首页）。
- `requirements.txt`

## 常用命令

```bash
pip install -r requirements.txt
python action_nca.py                          # 训练确定性和隐变量两个模型并评估，单核 CPU 约 11 分钟
python action_nca.py --device cuda            # 用 GPU
python sweep_steps.py --model nca_out/nca_latent.pt
python diffusion_baseline.py --device mps --train_only   # DP 基线在 Apple GPU 上训练，约 8 分钟
python diffusion_baseline.py --load dp_out/dp_policy.pt  # 在 CPU 上评估，并和 nca_out/ 里的 NCA 对比
python diffusion_baseline.py --method flow --device mps --train_only --out dp_out/flow   # 流匹配版本
python multi_seed.py                                     # 多种子实验，约 1 小时；中断后重跑会跳过已完成的步骤
```

macOS 上 PyTorch 的 NNPACK 一维卷积很慢，`diffusion_baseline.py` 里已经关掉了；U-Net 在 CPU 上训练仍然慢（大模型约 0.4 秒一轮），训练用 `--device mps`。

新的运行输出默认写到 `nca_out/`（已加入 .gitignore）。

## 代码结构（action_nca.py）

- `make_demos`、`chunk_targets`：二维玩具任务的示范和动作块目标。
- `ActionNCA`：`seed()` 种子细胞（当前位置 + z + 标记位），`blank()` 冷启动时“原地不动”的计划，`step()` 一步更新（固定核感知 → 拼上时间位置和观测编码 → 共享 MLP → 随机一半细胞更新），`grow()` 重复若干步。
- `shift()`：再生前切掉已执行的细胞、前移、补空白尾巴。
- `train()`：每一轮 = 冷启动样本（min-of-K 选 z）+ 样本池里的再生样本，两个误差相加。
  可选：`--prec_weight`（精度损失）、`--push_frac`（推动训练：放圆、边长边推、目标换成绕开圆的变形示范，见 `sample_circles`、`deform_targets`、`grow_pushed`）。
- `box_sdf`、`plan_cost`、`push`：避障代价和推动，`guidance.py`、`novel_obstacle.py` 共用。
- `run_episodes()`：闭环评估，`cold_fresh`、`cold_fixed`、`warm` 三种执行方式。
- `first_plan_offsets()`、`growth_frames()`、`plot()`：诊断和画图。

## 约定

- 用中文交流；代码注释用中文，标识符用英文。
- 保持单文件、易读的原型风格。新实验优先加命令行参数或新脚本，不要破坏现有的默认流程。
- 不要覆盖 `results/` 里的基线结果，除非明确要求更新。
- 汇报结果时写清训练轮数、评估回合数和随机种子；只有一个种子时要说明结论可能不稳。
- 结论要实事求是：预测和结果不符时直接说明（NOTES.md 里就有一个被结果修正的预测）。

## 下一步（按优先级）

1. ~~Diffusion Policy 基线~~：已完成（`diffusion_baseline.py`，结果见 NOTES.md 第 9 节）。DP 在成功率和算力上都不输 NCA，NCA 只在计划翻转（模态一致性）上明显更好。
2. ~~多种子~~：已完成（`multi_seed.py`，结果见 NOTES.md 第 11 节）。第 9 节的结论在 5 个种子上都成立；流匹配还没有做多种子。
3. ~~随机化障碍、让策略看见障碍~~：已完成（`random_obstacle.py`，NOTES.md 第 17 节）。目标位置还没有随机化。
4. 迁移到带图像观测的 Push-T。
