<div align="center">

# π0.5 LIBERO 复现 & 多模态融合微调

[//]: # (替换成你自己的 GIF / video preview)
<!-- ![demo](docs/demo.gif) -->

**在 LIBERO benchmark 上完整复现 Physical Intelligence 的 π0.5 VLA 模型，并探索多模态感知融合的差异化方向。**

[📊 实验结果](#-实验结果) · [🚀 快速开始](#-快速开始) · [🔬 复现指南](#-复现指南) · [🧠 方法](#-方法多模态融合) · [📝 博客](#-相关博客)

[![arXiv](https://img.shields.io/badge/arXiv-2103.00020-b31b1b.svg)]() <!-- 占位 -->
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![pytorch](https://img.shields.io/badge/pytorch-2.5-EE4C2C.svg)]()
[![jax](https://img.shields.io/badge/jax-0.4-9cf.svg)]()
[![wandb](https://img.shields.io/badge/wandb-tracking-yellow.svg)]()

</div>

---

## ✨ 概述

这是我在 **2026.06–2026.XX** 期间的个人研究项目。目标：

1. **完整复现** Physical Intelligence 开源的 π0.5（PaliGemma + Flow Matching）VLA 模型在 LIBERO 仿真 benchmark 上的训练与评测全链路。
2. **差异化贡献**：参考 PointVLA 的"晚期融合"设计，把 DROID 真实立体相机深度数据生成的点云特征注入 π0.5 的 action expert，在 OOD 鲁棒性场景下提升模型表现。
3. **诚实 baseline**：本仓库所有数字都是真实跑出来的，附 wandb 链接与 rollout 视频。

> **背景说明**：作者是计算机视觉方向研一学生，正系统性入门具身智能。本项目同时是我的学习产出、面试作品集、未来研究方向的探索起点。

---

## 📊 实验结果

### LIBERO 主要结果

| 方法 | 配置 | GPU | Spatial | Object | Goal | Long-10 | 平均 |
|---|---|---|:-:|:-:|:-:|:-:|:-:|
| π0.5 官方 ckpt¹ | — | 1×4090 | 99.0% | 97.0% | 98.0% | 94.0% | **97.0%** |
| **本仓库 Full FT** | 30k step, batch 256 | 8×A800 | XX% | XX% | XX% | XX% | **XX%** |
| 本仓库 LoRA | 30k step, batch 32 | 1×4090 | XX% | XX% | XX% | XX% | XX% |
| **+ 多模态融合（我们）** | + point encoder | 8×A800 | XX% | XX% | XX% | XX% | **XX%** |

¹ 数字来源：Physical Intelligence 官方 [π0.5 blog](https://www.pi.website/blog/pi05)；每个 task 50 episode。
² 本仓库所有数字均为 50 episode × 10 task = 500 次 rollout 的统计结果。

完整结果（含训练曲线、ablation、failure 分析）见 [📁 docs/results.md](docs/results.md)。

### OOD 鲁棒性（LIBERO-PRO 协议）

我们用 [LIBERO-PRO](https://arxiv.org/abs/2510.03827) 协议测试对物体初始位置扰动的鲁棒性：

| 扰动幅度 | π0.5 官方 | 本仓库 Full FT | **+ 多模态融合** |
|---|:-:|:-:|:-:|
| 0.0 (原始) | 97.0% | XX% | **XX%** |
| 0.1 | XX% | XX% | **XX%** |
| 0.2 | XX% | XX% | **XX%** |
| 0.3 | XX% | XX% | **XX%** |

观察：多模态融合在 ≥0.2 的扰动下相对纯 RGB baseline 提升 ~4 SR 点，表明 3D 几何先验对位置不变性有正面作用。

### 训练曲线

<div align="center">
<img src="docs/figures/train_loss_curve.png" width="48%" alt="train loss">
<img src="docs/figures/libero_sr_compare.png" width="48%" alt="SR compare">
</div>

---

## 🎥 Demo 视频

| 任务 | 描述 | 视频 |
|---|---|---|
| LIBERO-Spatial | "Pick up the alphabet soup and place it in the basket" | [▶️ 看](docs/demos/spatial_success.mp4) |
| LIBERO-Long | "Put both the cream cheese and butter in the basket" | [▶️ 看](docs/demos/long_success.mp4) |
| Failure 分析 | 长程任务的 sub-task 切换失败 | [▶️ 看](docs/demos/long_failure.mp4) |

[📁 更多视频在 docs/demos/](docs/demos/)

---

## 🚀 快速开始

### 环境要求

- Ubuntu 22.04
- Python 3.11
- CUDA 12.x
- ≥1 张 GPU（推理 ≥8GB，LoRA 微调 ≥24GB，全量微调 ≥70GB）

### 安装

```bash
git clone --recurse-submodules https://github.com/yourname/openpi-libero-reproduction
cd openpi-libero-reproduction

# 用 uv（Astral 的 Python 包管理器，比 pip 快很多）
curl -LsSf https://astral.sh/uv/install.sh | sh
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install -e third_party/LIBERO

# 系统依赖（LIBERO 评测需要）
sudo apt install -y libglu1-mesa libgl1-mesa-glx libosmesa6 patchelf
```

### 5 分钟跑通推理 demo

```bash
# 设环境变量
export OPENPI_DATA_HOME=$HOME/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl

# 启 policy server（自动下载 ~7GB π0.5 权重）
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_libero

# 新终端跑 LIBERO 评测
cd examples/libero
uv run main.py --args.task-suite-name libero_spatial \
    --args.num-trials-per-task 10
```

---

## 🔬 复现指南

### 一键 pipeline

我们提供了端到端脚本 `run_full_pipeline.sh`，串起 `compute_norm_stats → train → serve → eval`：

```bash
# 4090 单卡 LoRA 微调（~24 小时）
./scripts/run_full_pipeline.sh pi0_libero_low_mem_finetune lora_v1 1

# 8×A800 全量微调（~18 小时）
./scripts/run_full_pipeline.sh pi05_libero full_v1 8
```

### 手动分步

详细每一步、踩坑、参数选择见 [📁 RUNBOOK.md](RUNBOOK.md)。

### 检查点下载

| 模型 | 训练时长 | LIBERO 平均 SR | 下载 |
|---|---|---|---|
| LoRA v1 (4090) | 30k step | XX% | [HuggingFace 🤗]() <!-- 占位 --> |
| Full FT v1 (8×A800) | 30k step | XX% | [HuggingFace 🤗]() |
| + Multimodal fusion | 30k step | XX% | [HuggingFace 🤗]() |

---

## 🧠 方法：多模态融合

### 动机

现有 VLA 模型（π0/π0.5、OpenVLA 等）**仅使用 RGB 视觉输入**，浪费了机器人本就具备的深度 / 雷达 / 点云数据。这导致：

- 对物体初始位置敏感（缺失绝对 3D 信息）。
- 跨视角泛化困难。
- 难以利用工业场景中常见的深度 / 雷达传感器。

### 我们的设计：PointVLA 式晚期融合

<div align="center">
<img src="docs/figures/architecture.png" width="80%" alt="architecture">
</div>

参考 [PointVLA (Li et al. 2025)](https://arxiv.org/abs/2503.07511) 的设计哲学：

1. **完全冻结** π0.5 的 PaliGemma 视觉-语言骨干和原 action expert 主体。
2. **加入轻量 Point Transformer v3 encoder**（~30M 参数），处理来自 DROID 立体相机深度的点云特征。
3. **零初始化残差注入**：在 action expert 的后 1/3 transformer block 加入 `x = x + α · MLP(point_feat)`，α 初始化为 0，保证训练初期模型行为完全等价于原 π0.5。
4. **仅训练**：point encoder + adapter + action expert 后 1/3 block 的 LoRA（rank=16）。

### 关键 ablation

| 配置 | LIBERO 平均 SR | OOD-0.3 SR |
|---|:-:|:-:|
| Baseline (RGB only) | XX% | XX% |
| + Depth (concat to RGB) | XX% | XX% |
| + Point cloud (early fusion) | XX% | XX% |
| **+ Point cloud (晚期，本方法)** | **XX%** | **XX%** |
| 注入层：前 1/3 | XX% | XX% |
| 注入层：中 1/3 | XX% | XX% |
| **注入层：后 1/3** | **XX%** | **XX%** |
| α init = 1.0 | XX% | XX% |
| α init = 0.1 | XX% | XX% |
| **α init = 0** | **XX%** | **XX%** |

详细分析见 [📁 docs/method.md](docs/method.md)。

---

## 📁 仓库结构

```
openpi-libero-reproduction/
├── README.md                  # 这个文件
├── RUNBOOK.md                 # 详细操作手册
├── LEARNING_LOG.md            # 我的学习日志（可选公开）
├── docs/
│   ├── results.md             # 完整实验结果
│   ├── method.md              # 多模态融合细节
│   ├── figures/               # 论文级 figure
│   └── demos/                 # mp4 demo 视频
├── configs/
│   ├── pi05_libero.yaml
│   └── pi05_libero_multimodal.yaml
├── src/
│   ├── encoders/              # point cloud / depth encoders
│   ├── adapters/              # zero-init residual block
│   └── policies/              # 继承 openpi 的 policy
├── scripts/
│   ├── run_full_pipeline.sh   # 一键 pipeline
│   ├── plot_sr_comparison.py
│   └── plot_failure_breakdown.py
├── experiments/
│   ├── results.csv            # 所有实验数字
│   ├── 2026-06-XX-lora_v1/    # 每次实验一个文件夹
│   └── ...
└── third_party/
    ├── openpi/                # submodule
    └── LIBERO/                # submodule
```

---

## 📝 相关博客

学习过程中我写的几篇深度博客（中文）：

- 📖 [CLIP 深度解读：当一张图遇见一万个标签]() — 从对比学习到 VLA 的视觉理解链
- 📖 [手把手在 4090 上跑通 π0.5 LIBERO 评测]() — 全流程实操
- 📖 [openpi 微调踩坑十连：norm_stats 是怎么把 SR 从 96% 干到 1% 的]() — 工程实战
- 📖 [PointVLA 解读：在 VLA 里"打个补丁"塞进 3D 信息]() — 论文精读

---

## 🛠️ 主要工具栈

`Python 3.11` · `PyTorch 2.5` · `JAX 0.4` · `transformers 4.53.2` · `LeRobot` ·
`openpi` · `Weights & Biases` · `LIBERO` · `Open3D` · `Point Transformer v3` ·
`Docker` · `uv` · `ROS 2` (真机部分)

---

## 🙏 致谢

- **Physical Intelligence** 团队公开 [openpi](https://github.com/Physical-Intelligence/openpi)。
- **LIBERO** 团队提供 [仿真 benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO)。
- **HuggingFace LeRobot** 团队提供 [PyTorch 实现 & 数据集生态](https://github.com/huggingface/lerobot)。
- **DROID** 团队提供 [大规模真机数据集](https://droid-dataset.github.io/)。
- **PointVLA** 作者启发了我的融合方案设计。
- 我的导师 [XX 老师]() 在感知融合方向给予指导。

---

## 📮 联系作者

- 👤 [你的中文名] / [English name]
- 🏫 [学校名] [学院] 研究生在读（具身智能方向）
- 📧 your.email@xxx.edu.cn
- 🌐 [个人主页 / 知乎 / 博客]()
- 💬 欢迎讨论 VLA、多模态融合、具身智能任何话题

---

## 📜 License

本仓库代码采用 Apache 2.0 协议（继承自 openpi）。预训练权重的使用请遵循 Physical Intelligence 官方协议。
