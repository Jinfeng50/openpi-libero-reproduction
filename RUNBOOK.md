# π0.5 复现微调操作手册（RUNBOOK · A800 版）

> 适配：8×A800-80GB 共享服务器（NV8 NVLink 全互联），日常主用 1-2 卡。
> 基础复现已完成；当前创新方向为训练免费的 DGTE 时序动作融合，正式 A/B 结果待 GPU 空闲后运行。
> 最后更新：2026-08。

---

## 0. 路径与硬件约定

| 名称 | 路径 / 配置 |
|---|---|
| openpi 项目根 | `/cfsdata/chenjinfeng/projects/openpi` |
| LIBERO 项目根 | `/cfsdata/chenjinfeng/projects/LIBERO` |
| lerobot 项目根 | `/cfsdata/chenjinfeng/projects/lerobot` |
| 个人项目根 | `/cfsdata/chenjinfeng/projects/openpi-libero-reproduction` |
| 数据集根 | `/cfsdata/chenjinfeng/datasets` |
| LIBERO 数据 | `/cfsdata/chenjinfeng/datasets/libero` |
| DROID 数据 | `/cfsdata/chenjinfeng/datasets/droid`（3.4T） |
| 本地权重根 | `/cfsdata/chenjinfeng/models/openpi`（pi0_base / pi0_fast_base / pi05_base） |
| openpi 缓存 | `/cfsdata/chenjinfeng/openpi_cache` |
| GPU 资源 | 8×A800-SXM4-80GB，**共享 39 用户**，日常稳定能用 1-2 张 |

### 关键硬件事实

- **每张 A800 显存 80GB** → 单卡就能放下 π0.5 全量微调，**不需要 LoRA**
- **NV8 NVLink 全互联** → 多卡 FSDP 通信效率高
- **NUMA 拓扑**：GPU 0-3 在 NUMA 0（CPU 0-61），GPU 4-7 在 NUMA 1（CPU 62-123）
- **1.8 TB 系统内存** → 数据加载完全不会瓶颈
- **`/cfsdata` 是 1PB 共享 NFS**，所有数据 / 权重 / 缓存都放这里
- **⚠️ 根分区 `/` 只有 200G 且 98% 满**，绝不能往 `~/` 或 `/tmp` 放任何大文件
- **transformers 锁死 4.53.2**，openpi 必须这个版本

---

## 1. 环境变量（写进 `~/.bashrc`，一次配好）

```bash
cat >> ~/.bashrc <<'EOF'

# === openpi project env vars ===
export OPENPI_DATA_HOME=/cfsdata/chenjinfeng/openpi_cache
export HF_HOME=/cfsdata/chenjinfeng/hf_cache
export HF_LEROBOT_HOME=/cfsdata/chenjinfeng/datasets
export TMPDIR=/cfsdata/chenjinfeng/tmp
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl
export GIT_LFS_SKIP_SMUDGE=1
export WANDB_PROJECT=openpi-libero
EOF

source ~/.bashrc
mkdir -p $TMPDIR $HF_HOME $OPENPI_DATA_HOME
```

---

## 2. 一次性初始化（新工作站做一次）

### 2.1 初始化 openpi submodules

⚠️ 你的环境快照显示 submodules 没初始化（`status` 输出前面带 `-`），必须先做：

```bash
cd /cfsdata/chenjinfeng/projects/openpi
git submodule update --init --recursive
git submodule status   # 验证：commit hash 前不再有 `-`
```

### 2.2 软链本地权重

```bash
mkdir -p $OPENPI_DATA_HOME/openpi-assets/checkpoints
for m in pi0_base pi0_fast_base pi05_base; do
    ln -sf /cfsdata/chenjinfeng/models/openpi/$m \
           $OPENPI_DATA_HOME/openpi-assets/checkpoints/$m
done
ls -la $OPENPI_DATA_HOME/openpi-assets/checkpoints/
```

### 2.3 验证 openpi 能识别本地权重

```bash
cd /cfsdata/chenjinfeng/projects/openpi
uv run python -c "
from openpi.shared import download
print('pi05_base ->', download.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))
"
# 期望秒返回，输出本地路径
```

### 2.4 LIBERO 数据软链

```bash
mkdir -p /cfsdata/chenjinfeng/datasets/physical-intelligence
ln -sf /cfsdata/chenjinfeng/datasets/libero \
       /cfsdata/chenjinfeng/datasets/physical-intelligence/libero
```

### 2.5 装 LIBERO 评测依赖

```bash
cd /cfsdata/chenjinfeng/projects/openpi
uv pip install -e /cfsdata/chenjinfeng/projects/LIBERO
```

### 2.6 注册 wandb

```bash
uv run wandb login
```

---

## 3. 健康检查（每次开实验前 2 分钟）

```bash
cd /cfsdata/chenjinfeng/projects/openpi

# 1. 环境
uv run python -c "
import torch, jax, transformers
assert transformers.__version__ == '4.53.2', 'transformers 必须 4.53.2'
print(f'torch: {torch.__version__}, devices={torch.cuda.device_count()}')
print(f'jax:   {jax.__version__}, devices={len(jax.devices())}')
print(f'transformers: {transformers.__version__}')
"

# 2. 权重
ls $OPENPI_DATA_HOME/openpi-assets/checkpoints/

# 3. GPU 状态
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits

# 4. 磁盘（确保 / 还有空间）
df -h | grep -E "cfsdata|^/dev"
```

---

## 4. GPU 选择策略（共享集群核心技能）

### 4.1 一键看哪几张卡空闲

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
    sort -k2 -t',' -rn | \
    awk -F', ' '{ printf "GPU %s: %5d MiB free\n", $1, $2 }'
```

输出例：

```
GPU 5: 81148 MiB free        ← 完全空闲
GPU 6: 81032 MiB free        ← 完全空闲
GPU 4: 67547 MiB free        ← 大部分空闲
GPU 1:  8404 MiB free        ← 几乎满了
```

**判定**：`free > 75000 MiB` 视为可用（留 5GB 给别人波动）。

### 4.2 选卡

| 场景 | GPU 数 | 怎么选 |
|---|---|---|
| **默认（你的日常）** | 1 张 | 选 `free` 最大的那张 |
| **运气好** | 2 张 | 优先同 NUMA：(0,1) (2,3) (4,5) (6,7) |
| **极少数情况** | 4–8 张 | 先和队友打招呼 |

⚠️ 训练前**必须** `export CUDA_VISIBLE_DEVICES=X`，否则 JAX 默认抢所有可见卡。

---

## 5. Phase A：验证环境（30 分钟）

不跑 aloha demo（外网拉不到权重）。直接验证本地：

```bash
cd /cfsdata/chenjinfeng/projects/openpi

# 验证 pi05_libero config + 本地 pi05_base 权重可加载
uv run python -c "
from openpi.training import config as C
cfg = C.get_config('pi05_libero')
print('✅ config OK')
print('   weight_loader:', cfg.weight_loader)
"

# 验证 LIBERO 数据可加载
uv run python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('physical-intelligence/libero')
print(f'✅ LIBERO loaded: {len(ds)} frames, {ds.num_episodes} episodes')
"
```

两个 ✅ 都过就进 Phase B。

---

## 6. Phase B：下载 pi05_libero + 跑官方 baseline（半天）

简历上有"对照官方数字"才严谨。

### 6.1 测试 HTTPS 通不通

```bash
curl -I --max-time 10 \
    https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/
```

- `HTTP/2 200` → 方法 A
- 超时 / 403 → 方法 B

### 6.2 方法 A：HTTPS 列文件 + 并行下载

```bash
DEST=/cfsdata/chenjinfeng/models/openpi/pi05_libero
mkdir -p $DEST

# 用 GCS JSON API 列文件
curl -s "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_libero/&maxResults=1000" \
    | python3 -c "
import sys, json
for item in json.load(sys.stdin).get('items', []):
    print(f'https://storage.googleapis.com/openpi-assets/{item[\"name\"]}')
" > /tmp/pi05_libero_urls.txt

wc -l /tmp/pi05_libero_urls.txt   # 看一下有多少文件

# 并行下载
cat /tmp/pi05_libero_urls.txt | xargs -P 8 -I {} bash -c '
    url="{}"
    rel="${url#https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/}"
    mkdir -p "'$DEST'/$(dirname $rel)"
    wget -q -O "'$DEST'/$rel" "$url" && echo "✓ $rel" || echo "✗ $rel"
'

# 软链到 openpi 缓存
ln -sf $DEST $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
```

### 6.3 方法 B：本地下完 scp 上来

在能联外网的机器（你电脑 / 实验室小机器）：

```bash
pip install gsutil
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero ~/pi05_libero/

# 传到服务器
rsync -avh --progress ~/pi05_libero/ \
    chenjinfeng@服务器IP:/cfsdata/chenjinfeng/models/openpi/pi05_libero/

# 服务器上软链
ln -sf /cfsdata/chenjinfeng/models/openpi/pi05_libero \
       /cfsdata/chenjinfeng/openpi_cache/openpi-assets/checkpoints/pi05_libero
```

### 6.4 跑官方 baseline 评测

```bash
cd /cfsdata/chenjinfeng/projects/openpi

# 启 server（占 1 张空闲卡）
CUDA_VISIBLE_DEVICES=5 uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
```

新终端跑评测（每 task 10 episode 快速验证，~30 分钟）：

```bash
cd /cfsdata/chenjinfeng/projects/openpi/examples/libero
RESULTS=/cfsdata/chenjinfeng/projects/openpi-libero-reproduction/experiments/baseline_official
mkdir -p $RESULTS

for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== $SUITE ==="
    uv run main.py \
        --args.task-suite-name $SUITE \
        --args.num-trials-per-task 10 \
        --args.video-out-path /cfsdata/chenjinfeng/datasets/eval_videos/baseline_official/$SUITE \
        2>&1 | tee $RESULTS/eval_${SUITE}.log
done

grep -h "Success rate" $RESULTS/eval_*.log | tee $RESULTS/sr_summary.txt
```

期望 SR：

| Suite | 官方报告 | 你大概复现到 |
|---|---|---|
| Spatial | ~99% | 88–98% |
| Object | ~97% | 85–96% |
| Goal | ~98% | 85–96% |
| Long-10 | ~94% | 70–90% |

如果差 >15%，几乎一定是 norm_stats 错配。

---

## 7. Phase C：数据 + norm_stats（10 分钟）

```bash
cd /cfsdata/chenjinfeng/projects/openpi

# 看 config 从哪个 base 加载
uv run python -c "
from openpi.training import config as C
cfg = C.get_config('pi05_libero')
print('Weight loader:', cfg.weight_loader)
"

# 计算 norm_stats（仅用于从 base 微调）
CUDA_VISIBLE_DEVICES=5 uv run scripts/compute_norm_stats.py --config-name pi05_libero
ls assets/pi05_libero/
```

**⚠️ 核心陷阱**：
- 从 `pi05_base` **微调**时 → 用刚算的 norm_stats
- 评测**官方 `pi05_libero`** 时 → 必须用 ckpt 自带的 norm_stats

---

## 8. Phase D：训练（你的核心工作）

**默认场景**：1 张 A800-80GB，全量微调（80GB 显存够，不再需要 LoRA）。

### 8.1 单卡训练（日常）

30k step on 1×A800 估计 **~30-40 小时**。

```bash
cd /cfsdata/chenjinfeng/projects/openpi

# 先看哪张卡空
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -t',' -rn | head -3

# 设实验名
export EXP_NAME=pi05_libero_1gpu_$(date +%Y%m%d_%H%M)

# 启动训练（用 tmux！）
tmux new -s train

# tmux 内：
CUDA_VISIBLE_DEVICES=5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_libero \
    --exp-name=$EXP_NAME \
    --overwrite \
    --batch-size 32 \
    --num-train-steps 30000 \
    --save-interval 5000

# Ctrl+B 然后 D 离开
# 回来看：tmux attach -t train
```

### 8.2 2 卡训练（运气好或协调到）

本次 30k step on 2×A800 的实测墙钟时间约 **28.4 小时**（含最终 checkpoint 保存）；早期 16-20 小时只是排期估计，不作为承诺。

```bash
export EXP_NAME=pi05_libero_2gpu_$(date +%Y%m%d_%H%M)

# 优先同 NUMA：(4,5) 都在 NUMA 1
CUDA_VISIBLE_DEVICES=4,5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_libero \
    --exp-name=$EXP_NAME \
    --overwrite \
    --fsdp-devices 2 \
    --batch-size 64 \
    --num-train-steps 30000 \
    --save-interval 5000
```

### 8.3 4 卡 / 8 卡（少见，需协调）

```bash
# 4 卡，同 NUMA
CUDA_VISIBLE_DEVICES=4,5,6,7 \
uv run scripts/train.py pi05_libero \
    --exp-name=pi05_libero_4gpu_$(date +%Y%m%d) \
    --overwrite --fsdp-devices 4 --batch-size 128 \
    --num-train-steps 30000 --save-interval 5000

# 8 卡满配
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run scripts/train.py pi05_libero \
    --exp-name=pi05_libero_8gpu_$(date +%Y%m%d) \
    --overwrite --fsdp-devices 8 --batch-size 256 \
    --num-train-steps 30000 --save-interval 5000
```

### 8.4 训练时长 / batch 速查

| 卡数 | Batch | 30k step 时长（估） | 备注 |
|---|---|---|---|
| 1 | 32 | ~30-40h | **你的日常** |
| 2 | 64 | ~28.4h（本次实测） | 含最终 checkpoint 保存 |
| 4 | 128 | ~9-12h | 运气好 |
| 8 | 256 | ~6-8h | 需协调 |

### 8.5 监控

wandb 浏览器看曲线：
- `train/loss`：本次记录从 0.0726（step 100）降至约 0.0110（step 29,900）
- `train/grad_norm`：本次后期约 0.05-0.07，未见持续爆炸
- `train/learning_rate`：warmup 后 cosine decay

服务器端：

```bash
watch -n 5 nvidia-smi
```

### 8.6 断点续训（共享集群必备）

卡被抢、训练中断、想换卡继续：

```bash
# 例：原 GPU 5，现在被占；换 GPU 6 续训
CUDA_VISIBLE_DEVICES=6 \
uv run scripts/train.py pi05_libero \
    --exp-name=$EXP_NAME \
    --resume
```

---

## 9. Phase E：评测自己微调出来的 ckpt

```bash
cd /cfsdata/chenjinfeng/projects/openpi
CKPT=/cfsdata/chenjinfeng/projects/openpi/checkpoints/pi05_libero/$EXP_NAME/29999

# 启 server
CUDA_VISIBLE_DEVICES=5 uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=$CKPT
```

新终端跑正式评测（每 task 50 episode，~5-8 小时）：

```bash
cd /cfsdata/chenjinfeng/projects/openpi/examples/libero
RESULTS=/cfsdata/chenjinfeng/projects/openpi-libero-reproduction/experiments/$EXP_NAME
mkdir -p $RESULTS

for SUITE in libero_spatial libero_object libero_goal libero_10; do
    echo "=== Evaluating $SUITE ==="
    uv run main.py \
        --args.task-suite-name $SUITE \
        --args.num-trials-per-task 50 \
        --args.video-out-path /cfsdata/chenjinfeng/datasets/eval_videos/$EXP_NAME/$SUITE \
        2>&1 | tee $RESULTS/eval_${SUITE}.log
done

grep -h "Total success rate:" $RESULTS/eval_*.log | tee $RESULTS/sr_summary.txt
```

### 9.1 中间快速验证（挑最佳 ckpt 用）

```bash
# 跑 10 episode 看哪个 step 最好
for STEP in 10000 15000 20000 25000 30000; do
    CKPT=/cfsdata/chenjinfeng/projects/openpi/checkpoints/pi05_libero/$EXP_NAME/$STEP
    echo "=== Testing step $STEP ==="
    # 重启 server 指向新 step → 跑 10 episode → 记录
done
```

---

## 10. 可视化

### 10.1 SR 三方对比图

```bash
cd /cfsdata/chenjinfeng/projects/openpi-libero-reproduction
/cfsdata/chenjinfeng/projects/openpi/.venv/bin/python scripts/plot_results.py
```

脚本读取 `experiments/results.csv` 中已完成的真实记录，生成
`docs/figures/libero_sr_comparison.png`；不再在文档中内嵌未验证的示例数字。

### 10.2 训练曲线

```python
import wandb
api = wandb.Api()
run = api.run('3267189544-uestc/openpi/<RUN_ID>')
hist = run.history(keys=['train/loss', 'train/grad_norm', 'train/learning_rate'])
hist.to_csv('docs/figures/train_history.csv')

import matplotlib.pyplot as plt
plt.plot(hist['_step'], hist['train/loss'])
plt.xlabel('step'); plt.ylabel('loss')
plt.savefig('docs/figures/train_loss.png', dpi=150)
```

### 10.3 Demo 视频精选

```bash
mkdir -p ~/projects/openpi-libero-reproduction/docs/demos
# 挑 1 成功、1 边缘、1 失败
cp /cfsdata/chenjinfeng/datasets/eval_videos/$EXP_NAME/libero_10/task_0_seed_0.mp4 \
   ~/projects/openpi-libero-reproduction/docs/demos/long_success.mp4
```

---

## 11. 常见错误速查

| 现象 | 原因 | 解决 |
|---|---|---|
| 训练 loss 不降 | norm_stats 错配 | 从 base 微调用 compute_norm_stats 算的 |
| 评测 SR < 5% | norm_stats 错配 | 评测官方 ckpt 用 ckpt 自带 |
| `gsutil` 下载失败 | 外网不通 | 用本地权重 + HTTPS 下 pi05_libero |
| Submodule 报缺失 | 没初始化 | `git submodule update --init --recursive` |
| `libGLU.so.0 not found` | 缺 OpenGL | `apt install libglu1-mesa libgl1-mesa-glx`（需 sudo） |
| LIBERO 评测黑屏 | mujoco 没 EGL | `export MUJOCO_GL=egl` |
| JAX 占了所有 GPU | 没设 CUDA_VISIBLE_DEVICES | 训练前**必须**显式 export |
| 根分区写满 | 大文件落到 `~/` 或 `/tmp` | 所有 cache/tmp 走 `/cfsdata` |
| 训练突然 OOM | 别人抢内存 | tmux 中断 → 换 GPU `--resume` |
| Shape mismatch | base 错配 | pi05_libero config 必须配 pi05_base |
| `transformers != 4.53.2` | 严格版本 | `uv pip install transformers==4.53.2` |

---

## 12. 实验记录格式

`experiments/results.csv` 当前采用宽表，每行对应一次完整四-suite 评测：

```csv
date,exp_name,eval_run_id,checkpoint,spatial,object,goal,libero_10,average,episodes_per_suite,total_episodes
2026-06-12,pi05_libero_2gpu_20260610_1529,post_train_eval_20260612_1120,/cfsdata/chenjinfeng/projects/openpi/checkpoints/pi05_libero/pi05_libero_2gpu_20260610_1529/29999,0.984,0.984,0.968,0.918,0.9635,500,2000
```

---

## 13. 简历表述模板（填空）

```
基于 openpi 的 π0.5 VLA 模型复现 | 个人项目 | 2026.05 – 2026.08

· 完整复现 Physical Intelligence π0.5（PaliGemma + Flow Matching）VLA 模型，
  覆盖从 pre-trained base 到 LIBERO 仿真 benchmark 的全量微调与评测全链路。

· 在共享 A800-80GB 集群上完成 pi05_base → LIBERO 的 30k step 全量微调
  （batch=32-64，bf16，1-2 卡），LIBERO 4 个 suite（Spatial/Object/Goal/Long-10）
  分别达 SR 98.4% / 98.4% / 96.8% / 91.8%（共 2000 评测 episode），平均 96.35%；
  官方 checkpoint 独立复现平均 96.60%。

· 解决工程难题：norm_stats 错配导致 SR 崩塌、多人共享 GPU 调度策略、JAX 显存控制、
  transformers 严格版本依赖、MuJoCo headless 渲染、submodule 初始化、外网受限下
  权重分发等 7+ 个问题。

· 关键技术栈：VLA / π0.5 / Flow Matching / PaliGemma / FSDP / Bf16 / JAX 0.5 /
  PyTorch 2.7 / LeRobot / LIBERO / Weights & Biases。

· GitHub: github.com/Jinfeng50/openpi-libero-reproduction（含训练曲线、SR 结果、
  复现脚本和 DGTE 配对评测入口）。
```

---

## 14. DGTE 推理创新

当前已实现并测试 `src/openpi_libero_reproduction/temporal_ensemble.py`：
重叠 action chunk 的时序融合、分歧门控和夹爪离散保护。用下面的脚本做同一
checkpoint 的 baseline/DGTE 配对评测：

```bash
cd /cfsdata/chenjinfeng/projects/openpi-libero-reproduction
EXP_NAME=pi05_libero_2gpu_20260610_1529 GPU_ID=<空闲卡> \
N_EPISODES_PER_TASK=50 ./scripts/run_temporal_ablation.sh
```

10 episode/task 只用于 smoke test，正式结果必须写入 `experiments/<run>/sr_summary.csv` 后再更新 README。深度/点云/雷达融合暂不宣称结果，待数据和标定条件具备后另开实验。

---

## 15. 每日开工三件套

```bash
# 1. 进环境（变量已在 ~/.bashrc）
cd /cfsdata/chenjinfeng/projects/openpi-libero-reproduction

# 2. 看 GPU 状态
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -t',' -rn

# 3. 看昨天结果
tail -50 experiments/$(ls -t experiments/ 2>/dev/null | head -1)/eval_*.log 2>/dev/null
# 打开 wandb 网页端

# 4. 在 LEARNING_LOG.md 写今天计划
```
