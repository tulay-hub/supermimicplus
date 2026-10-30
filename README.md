# supermimicplus

Lens110 全身 21 维动作跟踪：在线学习可微世界模型，用短时想象轨迹的反向传播更新策略，并用价值网络估计远期回报。项目包含分阶段训练、153 条训练动作、关键检查点、完整评测，以及一条未参与训练的随机动作演示。

[![未见动作完整动图演示](docs/assets/unseen_cmu_91_04.gif)](https://github.com/tulay-hub/supermimicplus/releases/download/v0.1.0-30000/unseen_cmu_91_04.mp4)

**上方 GIF 可直接观看完整动作；[打开高清 MP4 视频](https://github.com/tulay-hub/supermimicplus/releases/download/v0.1.0-30000/unseen_cmu_91_04.mp4)。** 30000 轮模型，白色为策略控制的 Lens110，蓝色为参考。`cmu_91_04` 从 357 条排除训练集的候选中随机抽取，完整运行 18 秒，归一化回报 0.9188，未触发硬终止。保留真实全局位移，不把参考根位置对齐到机器人。

## 项目原理

项目学习的是一个**以参考动作作为目标的全身控制器**。每个控制周期，策略读取机器人最近 10 帧观测，以及接下来 10 帧参考动作，输出 21 个关节的控制动作。物理仿真执行动作后，策略继续根据新的机器人状态修正输出，因此播放的是闭环控制产生的轨迹。

训练由三个网络共同完成：

| 模块 | 输入和输出 | 在训练中的作用 |
|---|---|---|
| 策略网络 | 750D 观测历史 + 570D 未来参考 → 21D 动作 | 决定机器人下一步如何运动 |
| 世界模型 | 55D 状态特征 + 21D 动作 + 25D 仿真参数 → 54D 状态增量 | 学习动作会怎样改变状态，提供可微分的动力学预测 |
| 价值网络 | 机器人与参考的特征 → 未来回报估计 | 为有限想象窗口之后的表现提供估计 |

```mermaid
flowchart LR
    REF[未来参考动作] --> PI[目标条件策略]
    OBS[机器人观测历史] --> PI
    PI --> ACT[21 维关节动作]
    ACT --> SIM[PD 控制与物理仿真]
    SIM --> OBS
    SIM --> BUF[转移回放缓冲]
    BUF --> WM[世界模型学习]
    WM --> IMG[32 步可微想象轨迹]
    PI --> IMG
    IMG --> OBJ[跟踪奖励与价值估计]
    OBJ --> BPTT[反向传播更新策略]
    BPTT --> PI
```

**先从物理仿真学习动力学，再通过想象轨迹优化策略。** 采样器把状态、动作、仿真参数和下一状态存入回放缓冲，世界模型通过 8 步预测窗口拟合这些转移。策略训练时，世界模型展开 32 个控制步，把参考跟踪奖励和价值网络的远期估计组合成优化目标，再通过时间反向传播（BPTT）更新策略。

世界模型在联合训练阶段持续更新；每次策略反向传播期间固定世界模型参数，同时保留梯度通过预测状态、动作传到策略的路径。下一轮采样和世界模型训练继续使用新产生的物理数据。

实际关节目标为 `q_target = q_default + joint_offset + alpha * action`，由 PD 控制执行。双腿、腰和双臂均受同一个 21 维策略控制。奖励包含身体位置、姿态、线速度、角速度，以及根位置、根姿态六项指数跟踪，并加入动作变化和软关节限位惩罚。

未见动作演示使用冻结的策略和新的参考轨迹：从参考首帧初始化机器人后，由策略完整执行。演示结果用于观察参考条件下的动作泛化；姿态贴合程度和全局位移偏差需要分别检查。更详细的张量定义、奖励参数和模块关系见[算法与逻辑构成](docs/ARCHITECTURE.md)。

## 结果

训练于 2026-10-29 完成；RTX 5080 16 GB、4096 个环境、21 个受控关节。第二阶段 12000 → 30000 轮用时约 2 小时 47 分钟。全部训练阶段累计约 4 小时 37 分钟，不包含数据处理和评测。

| 30000 轮完整评测 | 完整执行 | 平均归一化回报 | 回报低于 0.82 |
|---|---:|---:|---:|
| 基础动作集 | 40/40 | 0.8477 | 8 |
| 扩充动作集 | 113/113 | 0.9204 | 5 |
| 全部训练动作 | 153/153 | 0.9014 | 13 |

原 40 条动作在 12000 轮的平均回报为 0.8340。全部动作按帧加权回报为 0.8569；短动作较多，所以同时保留按动作和按帧的口径。0/153 硬终止不等于零跟踪误差。以上为同仿真后端、每动作一次、无观测/动作噪声的完整参考回放，未包含扰动、跨仿真或实机验证。

[下载数据、模型和机器人资产](https://github.com/tulay-hub/supermimicplus/releases/tag/v0.1.0-30000)

## 安装与资源

从仓库根目录运行，推荐 Linux、Python 3.12、支持对应 PyTorch 的 NVIDIA 驱动。完整规模运行曾占用约 15.3 GB 显存。项目使用 MuJoCo 3.11、MuJoCo Warp、UniLab 1.5.1、UniSim 2.1.1；实际训练环境版本记录在 `reports/runtime_versions.json`。

```bash
git clone https://github.com/tulay-hub/supermimicplus.git
cd supermimicplus
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 先按显卡/驱动安装合适的 PyTorch，再安装项目。
python -m pip install -e . pytest ruff mypy pyright
python tools/fetch_release_assets.py
```

资源以 GitHub Release 附件分发，包含完整动作数据包、12000/30000 轮模型包和机器人资产包；下载器检查 SHA-256，并拒绝覆盖不同内容。代码仓库中的 `release-manifest.json` 记录每个附件的大小、校验和和内容。

## 分阶段训练

| 阶段 | 动作 | 轮次 | 更新内容 |
|---|---:|---:|---|
| 世界模型预训练 | 40 | 0–500 | 采集物理转移，更新世界模型；策略和价值网络尚不训练 |
| 基础联合训练 | 40 | 500–12000 | 持续更新世界模型、策略、价值网络 |
| 扩充联合训练 | 153 | 12000–30000 | 恢复完整检查点，加入 113 条动作，继续更新三个网络 |

以下启动命令用于新训练；附件已包含历史完成运行，脚本会拒绝覆盖。可先使用自定义的新目录；第二阶段通过 `SUPERMIMICPLUS_STAGE1_RUN` 指定第一阶段目录。

```bash
bash tools/train_lens110_lafan1.sh runs/my_stage1
SUPERMIMICPLUS_STAGE1_RUN=runs/my_stage1 \
  bash tools/train_lens110_expanded.sh runs/my_stage2
# 进程退出后恢复同一个第二阶段运行：
bash tools/train_lens110_expanded.sh runs/my_stage2 --resume
```

恢复完整的 `last.pt` 会保留三个网络、三个优化器和迭代计数；仿真状态与回放缓冲重新采集。编号 `ckpt_*.pt` 用于推理和评测，不能代替完整续训检查点。

## 评测与演示复现

```bash
supermimicplus-eval --backend mjwarp \
  --ckpt runs/lens110_lafan1_yahmp113_full21_30000_20261029/ckpt_030000.pt
MUJOCO_GL=egl python tools/preview_unseen.py
ffplay -autoexit outputs/unseen_cmu_91_04/unseen30000_cmu_91_04_meshes.mp4
```

未见动作脚本会先核对训练任务契约，仅替换参考文件及其哈希，并重新执行限位、速度、地面和自碰撞检查。测试参考不写回训练配置。

## 代码结构

```text
src/unisupermimic/
  conf/supermimic/                 # 算法、机器人、后端、153 动作配置
  scripts/train_supermimic.py      # 配置、运行目录、完整续训入口
  rl/algos/supermimic/             # replay、世界模型、策略、价值网络、BPTT
  tasks/motion_tracking/lens110/   # 21 关节、观测/目标、运动学和奖励契约
  tasks/motion_tracking/g1/supermimic/
                                  # 共享环境生命周期、动作、参考、评测和渲染
  sim2sim.py                      # 网络、机器人、动作库的严格契约检查
tools/                            # 训练、转换、审计、资源下载、未见动作播放
reports/                          # 两阶段日志、153 动作评测、未见动作与模型核对
docs/                             # 训练、架构、数据格式、演示视频
```

发行包名称与命令前缀是 `supermimicplus`。Python 内部包路径保留 `unisupermimic`，便于兼容已保存的配置和检查点；旧命令别名也保留。

详细说明：[分阶段训练](docs/TRAINING.md) · [算法和逻辑构成](docs/ARCHITECTURE.md) · [数据格式与打包内容](docs/DATASET.md) · [逐动作结果](reports/full153_eval_30000.json)。
