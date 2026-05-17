# π0 / π0.5 复现微调操作手册（RUNBOOK）

> 本文档是日常操作手册，跟项目代码同目录放置。每次开新终端、跑新实验、踩新坑都先翻这里。
>
> 最后更新：2026-05；适配 openpi 主分支。

---

## 0. 路径与机器约定

| 名称 | 路径 / 配置 |
|---|---|
| openpi 项目根 | `/cfsdata/chenjinfeng/projects/openpi` |
| LIBERO 项目根 | `/cfsdata/chenjinfeng/projects/LIBERO` |
| lerobot 项目根 | `/cfsdata/chenjinfeng/projects/lerobot` |
| 数据集根目录 | `/cfsdata/chenjinfeng/datasets` |
| LIBERO 数据 | `/cfsdata/chenjinfeng/datasets/libero` |
| DROID 数据 | `/cfsdata/chenjinfeng/datasets/droid` |
| OXE 数据 | `/cfsdata/chenjinfeng/datasets/oxe` |
| 本地权重根 | `/cfsdata/chenjinfeng/models/openpi` |
| openpi 缓存（symlink 后） | `/cfsdata/chenjinfeng/openpi_cache` |
| GPU 资源 | 偶发 8×A800-80GB + 常用 RTX 4090 |

---

## 1. 环境变量（每次新终端必跑，或写进 ~/.bashrc）

```bash
export OPENPI_DATA_HOME=/cfsdata/chenjinfeng/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl
export GIT_LFS_SKIP_SMUDGE=1
export HF_HOME=/cfsdata/chenjinfeng/hf_cache    # 让 HuggingFace 也用大盘
export WANDB_PROJECT=openpi-libero
```

---

## 2. 一次性初始化（只跑一次，跑过就跳过）

### 2.1 软链本地权重到 openpi 缓存

```bash
mkdir -p $OPENPI_DATA_HOME/openpi-assets/checkpoints
for m in pi0_base pi0_fast_base pi05_base; do
    ln -sf /cfsdata/chenjinfeng/models/openpi/$m \
           $OPENPI_DATA_HOME/openpi-assets/checkpoints/$m
done
ls -la $OPENPI_DATA_HOME/openpi-assets/checkpoints/
```

### 2.2 验证 openpi 能找到本地权重

```bash
cd /cfsdata/chenjinfeng/projects/openpi
uv run python -c "
from openpi.shared import download
p = download.maybe_download('gs://openpi-assets/checkpoints/pi05_base')
print('Resolved to:', p)
"
# 期望输出本地路径，且秒返回
```

### 2.3 注册 wandb（一次性）

```bash
uv run wandb login
```

### 2.4 装 LIBERO 评测依赖

```bash
cd /cfsdata/chenjinfeng/projects/openpi
uv pip install -e /cfsdata/chenjinfeng/projects/LIBERO
# 系统 OpenGL 依赖（如果是 root）
sudo apt install -y libglu1-mesa libgl1-mesa-glx libosmesa6 patchelf
```

---

## 3. 健康检查（每次大实验前跑一遍）

```bash
cd /cfsdata/chenjinfeng/projects/openpi

# 1. uv & openpi
uv --version
uv run python -c "import openpi; print(openpi.__file__)"

# 2. CUDA / JAX / PyTorch
uv run python -c "import torch; print('torch CUDA:', torch.cuda.is_available(), torch.cuda.device_count())"
uv run python -c "import jax; print('jax devices:', jax.devices())"

# 3. 数据集可读
ls /cfsdata/chenjinfeng/datasets/libero | head
ls $OPENPI_DATA_HOME/openpi-assets/checkpoints/

# 4. GPU 现状（避开别人在跑的卡）
nvidia-smi
```

---

## 4. Phase A：推理 demo（4 小时内出第一个胜利）

```bash
cd /cfsdata/chenjinfeng/projects/openpi

# 终端 1：启 policy server
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_aloha_pen_uncap \
    --policy.dir=gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap

# 终端 2：跑 sim client
cd /cfsdata/chenjinfeng/projects/openpi/examples/aloha_sim
uv run main.py
```

⚠️ 这一步用 ALOHA pen uncap ckpt（约 7GB），如果本地没有会从 gs:// 下载。可以提前 `ls $OPENPI_DATA_HOME/openpi-assets/checkpoints/` 看下有没有。如果服务器外网不通，跳过这步，直接用 `pi05_base` 跑训练验证。

---

## 5. Phase B（可选）：评测官方 pi05_libero baseline

**前提**：需要先下载 `pi05_libero` 权重（你目前只有 base，没有这个）。如果跳过，直接进 Phase C-D 自己微调。

```bash
# 下载 pi05_libero（约 7GB）
uv run python -c "
from openpi.shared import download
p = download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')
print(p)
"

# 启 server
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_libero

# 新终端跑评测
cd /cfsdata/chenjinfeng/projects/openpi/examples/libero
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    uv run main.py \
        --args.task-suite-name $SUITE \
        --args.num-trials-per-task 10 \
        --args.video-out-path /cfsdata/chenjinfeng/datasets/eval_videos/baseline_$SUITE
done
```

期望 SR（每 task 10 episode 时方差大，跑 50 episode 才稳）：

| Suite | 官方报告 | 你大概会得到 |
|---|---|---|
| Spatial | ~99% | 88–98% |
| Object | ~97% | 85–96% |
| Goal | ~98% | 85–96% |
| Long (10) | ~94% | 70–90% |

---

## 6. Phase C：LIBERO 数据预处理 + norm_stats

### 6.1 让 openpi 找到 LIBERO 数据

```bash
# 让 lerobot 标准路径指向你的数据
export HF_LEROBOT_HOME=/cfsdata/chenjinfeng/datasets
mkdir -p /cfsdata/chenjinfeng/datasets/physical-intelligence
ln -sf /cfsdata/chenjinfeng/datasets/libero \
       /cfsdata/chenjinfeng/datasets/physical-intelligence/libero
```

如果数据是 RLDS 原始格式，需要先转：

```bash
cd /cfsdata/chenjinfeng/projects/openpi
uv run examples/libero/convert_libero_data_to_lerobot.py \
    --data_dir /cfsdata/chenjinfeng/datasets/libero \
    --output_dir /cfsdata/chenjinfeng/datasets/libero_lerobot
```

### 6.2 看训练 config 期望的 base 权重

```bash
uv run python -c "
from openpi.training import config as C
for name in ['pi0_libero', 'pi0_libero_low_mem_finetune', 'pi05_libero']:
    try:
        cfg = C.get_config(name)
        print(f'{name}: weight_loader =', cfg.weight_loader)
    except Exception as e:
        print(f'{name}: {e}')
"
```

记下来：每个 config 默认从哪个 base 加载。**用 4090 LoRA 建议 `pi0_libero_low_mem_finetune`（从 pi0_base 加载，~22GB）**；用 A800 full FT 建议 `pi05_libero`（从 pi05_base 加载）。

### 6.3 计算 norm_stats

```bash
cd /cfsdata/chenjinfeng/projects/openpi
uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune
uv run scripts/compute_norm_stats.py --config-name pi05_libero
ls assets/
```

⚠️ **关键陷阱**：如果发现微调后 SR 极低（<10%），**99% 是 norm_stats 错配**。一个保险做法：保留两份 norm_stats，分别尝试。

---

## 7. Phase D：LoRA 微调（4090 单卡，约 24-30 小时）

```bash
cd /cfsdata/chenjinfeng/projects/openpi

export EXP_NAME=pi0_libero_lora_$(date +%Y%m%d)

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_libero_low_mem_finetune \
    --exp-name=$EXP_NAME \
    --overwrite \
    --num-train-steps 30000 \
    --batch-size 32 \
    --save-interval 5000 \
    --wandb-enabled
```

**实时监控**（开新终端）：

```bash
watch -n 5 nvidia-smi              # GPU 占用
tail -f /cfsdata/chenjinfeng/projects/openpi/logs/$EXP_NAME.log  # 训练日志
# wandb 网页端看 train/loss、train/grad_norm 曲线
```

**期望曲线**：
- `train/loss`：从 ~1.0 平稳下降到 0.2–0.4。
- `train/grad_norm`：稳定在 ~1.0（被 clip_grad_norm=1.0 截住）。
- `train/lr`：1000 step warmup 到 2.5e-5，之后 cosine decay。

**断点续训**（如果机器重启或 OOM）：

```bash
uv run scripts/train.py pi0_libero_low_mem_finetune --exp-name=$EXP_NAME --resume
```

---

## 8. Phase E：全量微调（8×A800，约 18-24 小时）

```bash
cd /cfsdata/chenjinfeng/projects/openpi

export EXP_NAME=pi05_libero_full_$(date +%Y%m%d)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_libero \
    --exp-name=$EXP_NAME \
    --overwrite \
    --fsdp-devices 8 \
    --batch-size 256 \
    --num-train-steps 30000 \
    --save-interval 5000 \
    --wandb-enabled
```

---

## 9. Phase F：评测微调后的 checkpoint

```bash
cd /cfsdata/chenjinfeng/projects/openpi

# 启 server，指向你训练出来的 ckpt
CKPT_DIR=/cfsdata/chenjinfeng/projects/openpi/checkpoints/$EXP_NAME/30000

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_libero_low_mem_finetune \
    --policy.dir=$CKPT_DIR

# 新终端跑评测（每 task 50 episode，正式数据）
cd /cfsdata/chenjinfeng/projects/openpi/examples/libero
RESULTS=/cfsdata/chenjinfeng/projects/openpi/experiments/$EXP_NAME
mkdir -p $RESULTS

for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== Evaluating $SUITE ==="
    uv run main.py \
        --args.task-suite-name $SUITE \
        --args.num-trials-per-task 50 \
        --args.video-out-path /cfsdata/chenjinfeng/datasets/eval_videos/$EXP_NAME/$SUITE \
        2>&1 | tee $RESULTS/eval_${SUITE}.log
done

# 把 SR 数字汇总（grep 找 "Success rate"）
grep -h "Success rate" $RESULTS/eval_*.log | tee $RESULTS/sr_summary.txt
```

把数字记进 `experiments/results.csv`。

---

## 10. 可视化（每个实验跑完都做）

### 10.1 SR 对比柱状图

```bash
cd /cfsdata/chenjinfeng/projects/openpi
uv run python scripts/plot_sr_comparison.py    # 自己写，参考下面
```

```python
# scripts/plot_sr_comparison.py
import matplotlib.pyplot as plt
import numpy as np

suites = ['Spatial', 'Object', 'Goal', 'Long']
baseline = [0.90, 0.92, 0.88, 0.76]    # 官方 ckpt（如果有跑）
my_lora  = [0.85, 0.88, 0.82, 0.65]    # 你的 LoRA
my_full  = [0.92, 0.93, 0.89, 0.78]    # 你的全量

x = np.arange(len(suites))
plt.figure(figsize=(8, 5))
plt.bar(x - 0.25, baseline, 0.25, label='π0.5 official')
plt.bar(x      , my_lora,   0.25, label='Mine - LoRA')
plt.bar(x + 0.25, my_full,  0.25, label='Mine - Full FT')
plt.xticks(x, suites)
plt.ylabel('Success Rate')
plt.title('LIBERO Reproduction Results')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/figures/libero_sr_compare.png', dpi=150)
```

### 10.2 Rollout 视频精选

每个 suite 挑 3 段：1 成功、1 边缘、1 失败：

```bash
mkdir -p experiments/demo_videos
# 在 eval_videos 里挑
cp /cfsdata/chenjinfeng/datasets/eval_videos/$EXP_NAME/libero_10/task_0_seed_0.mp4 \
   experiments/demo_videos/long_success.mp4
# ... 重复
```

### 10.3 Failure 分类表

人工分类，统计：

```python
# scripts/plot_failure_breakdown.py
import matplotlib.pyplot as plt

categories = ['Grasping', 'Placing', 'Subtask switch', 'Out of workspace', 'Other']
counts = [12, 8, 5, 3, 2]

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
plt.title('Failure Mode Breakdown (n=30)')
plt.tight_layout()
plt.savefig('experiments/figures/failure_breakdown.png', dpi=150)
```

---

## 11. 常见错误速查表

| 现象 | 原因 | 解决 |
|---|---|---|
| 训练 loss 一直 >1.0 不降 | norm_stats 错配 | 重算 norm_stats 或换用 base ckpt 自带版本 |
| 评测 SR < 5% | norm_stats 错配（同上） | 用训练时的 norm_stats，不要重算 |
| `XLA_PYTHON_CLIENT_MEM` 报错 | JAX 抢光显存 | 加 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` |
| `libGLU.so.0 not found` | 缺 OpenGL | `apt install libglu1-mesa libgl1-mesa-glx` |
| LIBERO 评测黑屏 | mujoco 没用 EGL | `export MUJOCO_GL=egl` |
| `transformers` AdaRMS 报错 | transformers 版本错 | 严格 `transformers==4.53.2` |
| `pi0_libero_low_mem_finetune` 占 44GB | 实际比 README 大 | 用 A100/A6000，或 4090 时关 EMA |
| Shape mismatch in weight loading | base ckpt 错配 | pi0 配 pi0_base，pi0.5 配 pi05_base |
| `git lfs` 卡死 | LFS smudge 拉大文件 | `GIT_LFS_SKIP_SMUDGE=1` 全程 export |
| `gs://...` 下载慢/失败 | 外网问题 | 提前 `maybe_download` 缓存，或软链本地权重 |

---

## 12. 实验记录格式（results.csv 表头）

```csv
date,exp_name,model,config,suite,n_episodes,success_rate,wandb_url,ckpt_path,notes
2026-06-15,baseline,pi05_libero_official,pi05_libero,spatial,50,0.94,,gs://.../pi05_libero,official
2026-06-20,lora_v1,pi05_libero_lora_30k,pi0_libero_low_mem_finetune,spatial,500,0.85,wandb.ai/xxx,ckpts/.../30000,30k step LoRA
```

---

## 13. 简历表述模板（跑完填空）

```
基于 openpi 框架的 π0.5 VLA 模型复现 | 个人项目 | 2026.06 – 2026.XX

· 完整复现 Physical Intelligence π0.5（PaliGemma + Flow Matching）VLA 模型
  从 base ckpt 到 LIBERO benchmark 的微调与评测全链路。

· 在 8×A800 GPU 集群上用 FSDP 完成 π0.5-base 到 LIBERO 的 30k step 全量
  微调（batch=256），LIBERO 4 个 task suite（Spatial/Object/Goal/Long-10）
  分别达 SR XX% / XX% / XX% / XX%（共 2000 个评测 episode），与官方差距 <X 点。

· 单卡 RTX 4090 上用 LoRA 完成同任务微调（22.5 GB 显存，30k step），平均
  SR XX%，验证低成本环境下 VLA 微调可行性。

· 解决工程难题：norm_stats 配置导致 SR 崩塌、JAX OOM 调参、transformers
  patch、MuJoCo headless 渲染、混合精度等 6+ 个问题。

· 技术栈：VLA / π0.5 / Flow Matching / PaliGemma / LoRA / FSDP / JAX /
  PyTorch / LeRobot / LIBERO / WandB。

· GitHub: github.com/[user]/openpi-libero-reproduction（训练曲线、SR 对比、
  Failure case、Rollout demo 视频齐全）。
```

---

## 14. 紧急情况

- **训练崩了不知道为什么**：先看 `experiments/$EXP_NAME/*.log` 最后 200 行；其次 `nvidia-smi` 看是否 OOM；再看 wandb 看 grad_norm 是否爆炸。
- **评测中途断了**：openpi `examples/libero/main.py` 不支持断点续跑，需要重启 server + 重跑评测；可以改小 `num-trials-per-task` 分批跑。
- **训练效果跟期望差很多**：先确认 norm_stats、再确认 base 权重对应、再确认数据是否完整（`len(dataset)` 应 >= 50 task × 50 demo × 200 step）。
- **A800 被抢了**：所有训练写 `--save-interval` 保 checkpoint，被抢时 `--resume` 续训。

---

## 15. 每日开工三件套

```bash
# 1. 进环境
cd /cfsdata/chenjinfeng/projects/openpi
source ~/.bashrc  # 确保环境变量在

# 2. 看昨天的实验
tail -50 experiments/$LAST_EXP/eval_*.log
# 看 wandb 的曲线

# 3. 决定今天要跑什么，把命令写进 LEARNING_LOG.md
```
