# π0.5 复现微调 + GitHub 开源全链路指南（A800 版）

> 作者：chenjinfeng
> 项目：基于 openpi 的 pi0.5 LIBERO 复现 + DGTE 时序动作融合
> 硬件：8×A800-80GB 共享服务器，日常主用 1-2 卡
> 适用：从 0 到秋招简历落地的全过程
> 最后更新：2026-05

---

## 0. 文档体系导航

四份文档分工：

| 文档 | 角色 | 何时看 | 公开 |
|---|---|---|---|
| **FULL_GUIDE.md** (本文) | 总指挥 | 项目开始 + 每个 Phase 之前 | ✅ |
| **RUNBOOK.md** | 技术手册 | 跑命令 + 报错时查 | ✅ |
| **run_full_pipeline.sh** | 一键脚本 | 跑实验时 | ✅ |
| **README.md** | 对外门面 | 有结果后才打磨 | ✅ |
| **LEARNING_LOG.md** | 私人记录 | 每天 5 分钟填 | ❌ |

---

## 0.1 日志保存约定

所有长命令都必须留日志。统一放到个人项目的 `experiments/<RUN_ID>/logs/` 下：

```bash
export PERSONAL_DIR=/cfsdata/chenjinfeng/projects/openpi-libero-reproduction
export RUN_ID=<阶段名>_$(date +%Y%m%d_%H%M)
export EXP_DIR=$PERSONAL_DIR/experiments/$RUN_ID
mkdir -p "$EXP_DIR/logs"
```

运行命令时统一使用：

```bash
some_command 2>&1 | tee "$EXP_DIR/logs/<stage>.log"
```

如果是训练这种长任务，日志文件名至少包含阶段和实验名，例如：

```text
experiments/pi05_libero_2gpu_20260610_1500/logs/02_train.log
```

这样以后回查时可以直接从 `experiments/` 知道：跑了什么、什么时候跑的、命令输出是什么、失败在哪里。

---

## Part 1：一次性准备（Day 0，4 小时）

### 1.1 GitHub 账号

- 注册：https://github.com
- 用户名：真名或类真名（`chenjinfeng` 这种），别用网名
- 用学校邮箱注册可申请 Student Pack（免费 Pro）：https://education.github.com/pack

### 1.2 Profile 完善

Settings → Public profile：
- **头像**：真人照
- **Bio**：`MS student · Embodied AI / VLA · Reproducing π0.5`
- **Location**：你所在城市

### 1.3 服务器 SSH（解决国内 22 端口被墙）

```bash
# 生成 key
ssh-keygen -t ed25519 -C "your.email@xxx.edu.cn"
cat ~/.ssh/id_ed25519.pub
# 复制到 GitHub → Settings → SSH and GPG keys → New SSH key
```

⚠️ 国内 22 端口经常被墙，**走 443**：

```bash
cat >> ~/.ssh/config <<'EOF'

Host github.com
  HostName ssh.github.com
  User git
  Port 443
  IdentityFile ~/.ssh/id_ed25519
EOF
chmod 600 ~/.ssh/config

ssh -T git@github.com
# 第一次问 yes/no 输入 yes
# 看到 "Hi Jinfeng50!" 就成功
```

### 1.4 git 配置

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@xxx.edu.cn"
git config --global init.defaultBranch main
git config --global pull.rebase false
```

### 1.5 创建主项目仓库

#### GitHub 网页：

1. `+` → New repository
2. 名称：`openpi-libero-reproduction`
3. 描述：`Reproducing Physical Intelligence's π0.5 VLA model on LIBERO + multimodal fusion exploration`
4. **Public**
5. **不要** 勾 README / .gitignore / license
6. Create

#### 本地建仓库 + push：

```bash
cd /cfsdata/chenjinfeng/projects
mkdir openpi-libero-reproduction && cd openpi-libero-reproduction

# 目录结构
mkdir -p src/{encoders,adapters,policies} \
         scripts configs docs/{figures,demos} \
         experiments notes third_party
touch experiments/.gitkeep notes/.gitkeep \
      docs/figures/.gitkeep docs/demos/.gitkeep

# 放四个文档（你已经有这些文件）
cp ~/RUNBOOK.md .
cp ~/README.md .
cp ~/FULL_GUIDE.md .
cp ~/run_full_pipeline.sh scripts/
chmod +x scripts/run_full_pipeline.sh

# .gitignore
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# Models & checkpoints
checkpoints/
*.ckpt
*.pt
*.pth
*.safetensors

# Data
data/
datasets/
*.h5
*.tfrecord*

# Experiment artifacts
experiments/*/
!experiments/.gitkeep
wandb/
runs/

# Logs
*.log
logs/

# Media (太大不进 repo)
*.mp4
*.mov
*.bag
*.zip

# Cache
.cache/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
*~

# Secrets
.env
*.key
*.pem
secrets/
EOF

# LICENSE
curl -s https://www.apache.org/licenses/LICENSE-2.0.txt > LICENSE

# submodules
git init
git submodule add https://github.com/Physical-Intelligence/openpi.git third_party/openpi
git submodule add https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO

# 首次 commit
git add .
git commit -m "Initial commit: project scaffold with RUNBOOK and pipeline script"

# 推 GitHub
git remote add origin git@github.com:Jinfeng50/openpi-libero-reproduction.git
git branch -M main
git push -u origin main
```

### 1.6 Profile README（个人主页置顶卡片）

```bash
cd /cfsdata/chenjinfeng/projects
mkdir Jinfeng50 && cd Jinfeng50

cat > README.md <<'EOF'
### Hi, I'm [你的中文名] 👋

CS MS student working on **embodied AI** / **vision-language-action models**.

🔬 **Current focus**: Reproducing π0.5 on LIBERO + exploring multimodal fusion.

📌 **Pinned**: [openpi-libero-reproduction](https://github.com/Jinfeng50/openpi-libero-reproduction)

📝 **Recent posts**:
- [CLIP 深度解读：当一张图遇见一万个标签](https://...)

📫 your.email@xxx.edu.cn
EOF

git init && git add . && git commit -m "Add profile README"
git remote add origin git@github.com:Jinfeng50/Jinfeng50.git
git branch -M main
git push -u origin main
```

### 1.7 Pin 仓库到主页

GitHub 主页 → Customize your pins → 把 `openpi-libero-reproduction` 钉到第一位。

### 1.8 Day 0 验收

- [x] GitHub 账号 + 头像 + bio
- [x] SSH 443 端口配通
- [x] `openpi-libero-reproduction` 公开 repo push 成功
- [x] Profile README 出现在主页
- [x] 主项目 pin 到首位

---

## Part 2：日常 Git 工作流

### 2.1 基本节奏

```bash
cd ~/projects/openpi-libero-reproduction

git status         # 看改了什么
git diff           # 看具体改了啥
git add .          # 暂存
git commit -m "..." # 提交
git push           # 推送
```

### 2.2 Commit message 规范（**HR 真的看**）

烂的（划走）：`update`、`fix`、`asdf`、`再次提交`、`修改了一下`

好的（英文，动词开头，≤72 字符）：

| 前缀 | 用途 | 示例 |
|---|---|---|
| `feat:` | 新功能 | `feat: add Point Transformer v3 encoder` |
| `fix:` | bug 修复 | `fix: correct norm_stats path causing SR collapse` |
| `docs:` | 文档 | `docs: add baseline reproduction results` |
| `exp:` | 实验 | `exp: 1×A800 30k step on LIBERO, avg SR=87%` |
| `refactor:` | 重构 | `refactor: extract eval loop into utility` |
| `chore:` | 杂项 | `chore: update .gitignore` |

### 2.3 频率

- 每完成一件可命名小事就 commit 一次
- 一天 3-8 个 commit 健康
- 绿格子图是工作量证明，**HR 看**

### 2.4 哪些"绝不能"push

- 密钥（.env、API key、SSH key、wandb token）
- 大文件（>50MB 的视频、ckpt）→ HuggingFace Hub
- 实验产物（checkpoints/、wandb/、*.log）→ 已在 .gitignore

### 2.5 Branch 策略

- 早期（M1-M2）：直接 push 到 `main`
- 多模态融合开发期：开 feature branch + PR

---

## Part 3：Phase A — 环境验证（Day 1，2 小时）

### 3.1 技术步骤

详见 `RUNBOOK.md` §1-§5。简版：

```bash
# 1. 环境变量进 ~/.bashrc（一次性）
# 2. 软链权重
# 3. submodule init
cd /cfsdata/chenjinfeng/projects/openpi
git submodule update --init --recursive

# 4. 验证
uv run python -c "
from openpi.training import config as C
cfg = C.get_config('pi05_libero')
print('✅', cfg.weight_loader)
"
```

### 3.2 Git 动作

```bash
cd ~/projects/openpi-libero-reproduction

# 把环境快照存到 docs（参考之前我让你做的 ~/check_env.sh）
~/check_env.sh    # 输出到 docs/env_snapshot.md

git add docs/env_snapshot.md
git commit -m "docs: add A800 environment snapshot"
git push
```

### 3.3 LEARNING_LOG entry

```markdown
### 2026-05-17 (Day 1)
**做了什么**：
- openpi 环境跑通，submodule init，本地权重软链。
- 第一个 GitHub repo openpi-libero-reproduction 公开。
- A800 集群 GPU 状态摸清楚：日常能稳定用 1-2 张。
**学到**：openpi 用 policy server + client 架构，websocket 通信。
**后续**：官方 checkpoint baseline 已完成，继续维护训练复现和 DGTE 配对评测。
```

### 3.4 Phase A 验收

- [x] openpi 环境 import + config 加载 OK
- [x] submodule 初始化完毕
- [x] `docs/env_snapshot.md` 推到 GitHub
- [x] LEARNING_LOG 写第一条

---

## Part 4：Phase B — 官方 checkpoint baseline 复测（Day 2-4）

### 4.1 官方到底怎么测

先把协议说清楚，否则数字没有可比性。

openpi 官方 LIBERO 入口是 `examples/libero/main.py`，默认：

- `task_suite_name="libero_spatial"`，命令行切换到 `libero_object` / `libero_goal` / `libero_10`
- `num_trials_per_task=50`
- `seed=7`
- `resize_size=224`
- `replan_steps=5`
- `num_steps_wait=10`
- 每个 suite 有 10 个 task，所以是 `10 task × 50 episodes = 500 episodes/suite`
- 四个 suite 合计 `2000 episodes`

openpi 官方推荐 Docker 跑 LIBERO eval，但服务器上你已经用非 Docker 环境跑通，所以当前复测采用“同一份官方 eval 脚本 + 官方默认参数 + 官方 ckpt”的非 Docker 流程。关键是：**不要传 `--args.num-trials-per-task 10`**，让脚本使用默认 50。

openpi 官方 README 里可对比的数字来自 `examples/libero/README.md`：

| Model | Spatial | Object | Goal | Libero 10 | Average |
|---|:-:|:-:|:-:|:-:|:-:|
| π0.5 @ 30k official ckpt | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |

本仓库已完成一次官方协议复现：

| Run | Spatial | Object | Goal | Libero 10 | Average |
|---|:-:|:-:|:-:|:-:|:-:|
| `baseline_official_50ep_20260607_1246` | 98.2 | 98.8 | 96.8 | 92.6 | 96.60 |

对应路径：

- Results: `/cfsdata/chenjinfeng/projects/openpi-libero-reproduction/experiments/baseline_official_50ep_20260607_1246`
- Videos: `/cfsdata/chenjinfeng/datasets/eval_videos/baseline_official_50ep_20260607_1246`

### 4.2 你上次为什么不能直接对比

上次 `baseline_official` 日志是真跑完了，但你传了 `--args.num-trials-per-task 10`，因此只有：

- 100 episodes/suite
- 四个 suite 合计 400 episodes
- 结果：Spatial 0.99 / Object 1.00 / Goal 0.97 / Libero10 0.94

这可以说明 policy、权重、LIBERO 环境基本健康，但不是官方 50 episodes/task 协议，不能写成最终 baseline。

### 4.3 复测前检查

```bash
cd /cfsdata/chenjinfeng/projects/openpi
export PERSONAL_DIR=/cfsdata/chenjinfeng/projects/openpi-libero-reproduction
export RUN_ID=baseline_precheck_$(date +%Y%m%d_%H%M)
export EXP_DIR=$PERSONAL_DIR/experiments/$RUN_ID
mkdir -p "$EXP_DIR/logs"

# 1. 确认官方 ckpt 本地存在
{
    ls -lah /cfsdata/chenjinfeng/models/openpi/pi05_libero
    ls -lah /cfsdata/chenjinfeng/openpi_cache/openpi-assets/checkpoints/pi05_libero
} 2>&1 | tee "$EXP_DIR/logs/01_checkpoint_paths.log"

# 2. 确认环境关键包
uv run python - <<'PY' 2>&1 | tee "$EXP_DIR/logs/02_python_env.log"
import transformers
print("transformers:", transformers.__version__)
assert transformers.__version__ == "4.53.2"
PY

# 3. 确认 config 能加载
uv run python - <<'PY' 2>&1 | tee "$EXP_DIR/logs/03_config.log"
from openpi.training import config as C
cfg = C.get_config("pi05_libero")
print("weight_loader:", cfg.weight_loader)
print("data:", cfg.data)
PY
```

### 4.4 启动官方 ckpt policy server

用 tmux 跑，避免 SSH 断了 server 也没了：

```bash
tmux new -s pi05_baseline_server
```

tmux 里执行：

```bash
cd /cfsdata/chenjinfeng/projects/openpi
export PERSONAL_DIR=/cfsdata/chenjinfeng/projects/openpi-libero-reproduction
export RUN_ID=baseline_server_$(date +%Y%m%d_%H%M)
export EXP_DIR=$PERSONAL_DIR/experiments/$RUN_ID
mkdir -p "$EXP_DIR/logs"

export OPENPI_DATA_HOME=/cfsdata/chenjinfeng/openpi_cache
export HF_HOME=/cfsdata/chenjinfeng/hf_cache
export HF_LEROBOT_HOME=/cfsdata/chenjinfeng/datasets
export TMPDIR=/cfsdata/chenjinfeng/tmp
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 选一张空卡；这里以 GPU 5 为例，运行前用 nvidia-smi 确认。
CUDA_VISIBLE_DEVICES=5 uv run scripts/serve_policy.py \
    --port 8001 \
    policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=/cfsdata/chenjinfeng/models/openpi/pi05_libero \
    2>&1 | tee "$EXP_DIR/logs/serve_policy.log"
```

看到 server 开始监听后，按 `Ctrl+B` 再按 `D` 退出 tmux。

### 4.5 官方协议完整评测（50 episodes/task）

新开一个 tmux：

```bash
tmux new -s pi05_baseline_eval
```

tmux 里执行：

```bash
cd /cfsdata/chenjinfeng/projects/openpi/examples/libero

export OPENPI_DATA_HOME=/cfsdata/chenjinfeng/openpi_cache
export HF_HOME=/cfsdata/chenjinfeng/hf_cache
export HF_LEROBOT_HOME=/cfsdata/chenjinfeng/datasets
export TMPDIR=/cfsdata/chenjinfeng/tmp
export MUJOCO_GL=egl
export PYTHONPATH=/cfsdata/chenjinfeng/projects/LIBERO:$PYTHONPATH

RUN_ID=baseline_official_50ep_$(date +%Y%m%d_%H%M)
RESULTS=/cfsdata/chenjinfeng/projects/openpi-libero-reproduction/experiments/$RUN_ID
VIDEOS=/cfsdata/chenjinfeng/datasets/eval_videos/$RUN_ID
mkdir -p "$RESULTS" "$VIDEOS"

for SUITE in libero_spatial libero_object libero_goal libero_10
do
    echo "=== Evaluating $SUITE with official 50 episodes/task protocol ==="
    uv run main.py \
        --args.task-suite-name "$SUITE" \
        --args.host=localhost \
        --args.port=8001 \
        --args.video-out-path "$VIDEOS/$SUITE" \
        2>&1 | tee "$RESULTS/eval_${SUITE}.log"
done

echo "suite,total_success_rate,total_episodes" > "$RESULTS/sr_summary.csv"
for SUITE in libero_spatial libero_object libero_goal libero_10
do
    LOG="$RESULTS/eval_${SUITE}.log"
    SR=$(grep "Total success rate:" "$LOG" | tail -1 | awk '{print $NF}')
    EP=$(grep "Total episodes:" "$LOG" | tail -1 | awk '{print $NF}')
    echo "$SUITE,$SR,$EP" >> "$RESULTS/sr_summary.csv"
done

cat "$RESULTS/sr_summary.csv"
```

注意：

- 这条命令没有 `--args.num-trials-per-task`，所以就是官方默认 50。
- 每个 suite 应该输出 `Total episodes: 500`。
- 四个 suite 预计 8-12 小时，务必 tmux。
- 退出时出现 `EGL_NOT_INITIALIZED` cleanup warning 通常不影响结果，只看 `Total success rate` 和 `Total episodes`。

### 4.6 评测完成后更新 `docs/baseline.md`

当前已完成的官方协议结果：

```csv
suite,total_success_rate,total_episodes
libero_spatial,0.982,500
libero_object,0.988,500
libero_goal,0.968,500
libero_10,0.926,500
```

评测完成后，把真实数字写进 `docs/baseline.md`：

```bash
cd /cfsdata/chenjinfeng/projects/openpi-libero-reproduction
mkdir -p docs

cat > docs/baseline.md <<'EOF'
# π0.5 LIBERO Official Checkpoint Baseline

## Protocol

Source: openpi `examples/libero/main.py` and `examples/libero/README.md`.

- Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`, mirrored locally at `/cfsdata/chenjinfeng/models/openpi/pi05_libero`
- Evaluation script: `examples/libero/main.py`
- Trials: 50 episodes per task
- Tasks: 10 tasks per suite
- Episodes: 500 episodes per suite, 2000 episodes total
- Seed: 7
- Replan steps: 5
- Image resize: 224

## Official Reference

| Model | Spatial | Object | Goal | Libero 10 | Average |
|---|:-:|:-:|:-:|:-:|:-:|
| π0.5 @ 30k official ckpt | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |

## My Previous Quick Check (Not Official Protocol)

This was run with 10 episodes per task, so it is only a sanity check.

| Suite | Episodes | Success Rate |
|---|---:|---:|
| Spatial | 100 | 99.0 |
| Object | 100 | 100.0 |
| Goal | 100 | 97.0 |
| Libero 10 | 100 | 94.0 |

## My Official-Protocol Reproduction

Completed run:

- Results: `/cfsdata/chenjinfeng/projects/openpi-libero-reproduction/experiments/baseline_official_50ep_20260607_1246`
- Videos: `/cfsdata/chenjinfeng/datasets/eval_videos/baseline_official_50ep_20260607_1246`

| Suite | Episodes | openpi Reference | Mine | Delta |
|---|---:|---:|---:|---:|
| Spatial | 500 | 98.8 | 98.2 | -0.6 |
| Object | 500 | 98.2 | 98.8 | +0.6 |
| Goal | 500 | 98.0 | 96.8 | -1.2 |
| Libero 10 | 500 | 92.4 | 92.6 | +0.2 |
| **Average** | 2000 | **96.85** | **96.60** | **-0.25** |

## Notes

- Do not pass `--args.num-trials-per-task 10` for official comparison.
- Evaluating official ckpt should use the checkpoint-bundled normalization stats.
- The final log line must contain `Total episodes: 500` for each suite.
- EGL cleanup warnings at process exit are harmless if total success rate and episode count were printed.
EOF

git add FULL_GUIDE.md docs/baseline.md
git commit -m "docs: record official pi05 LIBERO baseline reproduction"
git push
```

后续如果重跑 50ep 完整评测，再把新 run 和结果追加到 `docs/baseline.md`：

```bash
git add docs/baseline.md experiments/
git commit -m "exp: reproduce pi05 official LIBERO baseline with 50ep per task"
git push
```

### 4.7 第一篇博客

主题：**"手把手在共享 A800 集群上跑通 π0.5 LIBERO 评测"**

- 平台：知乎专栏 + CSDN 双发
- 长度：3000-5000 字
- 内容：环境配置、权重下载思路、norm_stats 坑、首组数字
- **发布后**：链接进 README 和 Profile README

### 4.8 Phase B 验收

- [x] `pi05_libero` 权重下载完成
- [x] 10ep 快速评测跑通（sanity check，SR 接近官方即可）
- [x] `docs/baseline.md` 建好并推到 GitHub（10ep 结果先填）
- [x] **50ep 官方协议评测跑完**（`baseline_official_50ep_20260607_1246`）
- [ ] README 主表格 "Official ckpt" 行填上 50ep 的真实数字
- [x] 第一篇博客发布
- [x] GitHub 累计 ≥10 commit

---

## Part 5：Phase C — 数据 + norm_stats（Day 5，30 分钟）

### 5.1 技术

详见 `RUNBOOK.md` §7。

`compute_norm_stats.py` 不是训练模型，而是在为“从 `pi05_base` 微调到 LIBERO”准备 state/action 的归一化统计量。可以把 `norm_stats` 理解成这个机器人数据集的“单位和尺度校准表”。

为什么需要它：

- LIBERO 的状态和动作维度尺度不同。末端位置可能是米制小数，姿态可能在 `-3` 到 `3`，夹爪动作又接近离散开合。
- 如果直接训练，数值范围大的维度会主导 loss，数值范围小但控制上很重要的维度可能学不好。
- 归一化会把不同维度转换到更接近的尺度，让优化更稳定。
- π0.5 是通用 VLA 架构，不同机器人和数据集的动作空间不同；`norm_stats` 告诉模型“当前这个数据集的 state/action 大概长什么样”。

两个场景一定要分清：

| 场景 | norm_stats 用法 |
|---|---|
| 评测官方 `pi05_libero` checkpoint | 用 checkpoint 自带的 norm_stats，不要重新算 |
| 从 `pi05_base` 微调到 LIBERO | 先对 LIBERO 训练数据计算 norm_stats，再训练 |

如果把这两种混用，动作反归一化会错，评测成功率可能直接崩掉。

```bash
cd /cfsdata/chenjinfeng/projects/openpi
export PERSONAL_DIR=/cfsdata/chenjinfeng/projects/openpi-libero-reproduction
export RUN_ID=norm_stats_pi05_libero_$(date +%Y%m%d_%H%M)
export EXP_DIR=$PERSONAL_DIR/experiments/$RUN_ID
mkdir -p "$EXP_DIR/logs"

CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name pi05_libero \
    2>&1 | tee "$EXP_DIR/logs/compute_norm_stats.log"

ls -lah /cfsdata/chenjinfeng/projects/openpi/assets/pi05_libero/physical-intelligence/libero/norm_stats.json \
    2>&1 | tee "$EXP_DIR/logs/norm_stats_file.log"
```

### 5.2 Git

5.1 已经生成：

```text
/cfsdata/chenjinfeng/projects/openpi/assets/pi05_libero/physical-intelligence/libero/norm_stats.json
```

当前 LIBERO 数据规模：

```text
frames:   273465
episodes: 1693
features: actions, state, image, wrist_image, task_index, episode_index, frame_index, timestamp, index
```

把这些真实信息写入 `docs/data.md`，不要再保留 `XX episodes / XX frames` 这种占位。

```bash
cd ~/projects/openpi-libero-reproduction

cat > docs/data.md <<'EOF'
# Datasets

## LIBERO
- Path: `/cfsdata/chenjinfeng/datasets/libero` (linked to `physical-intelligence/libero`)
- Format: LeRobotDataset
- Stats: 1693 episodes, 273465 frames, action_dim=7
- Features: actions, state, image, wrist_image, task_index, episode_index, frame_index, timestamp, index

## Generated norm_stats
- Path: `/cfsdata/chenjinfeng/projects/openpi/assets/pi05_libero/physical-intelligence/libero/norm_stats.json`
- Contains: state/actions mean, std, q01, q99
- State stats dim: 8
- Action stats dim: 7

## norm_stats Strategy
- Purpose: normalize LIBERO state/action dimensions so training sees stable numeric scales.
- From base fine-tuning (`pi05_base` → LIBERO): use freshly computed stats from `compute_norm_stats.py`.
- Evaluating official `pi05_libero` checkpoint: use checkpoint-bundled stats.
- Never mix these two cases; wrong stats distort action denormalization and can collapse SR.
EOF

git add docs/data.md
git commit -m "docs: dataset organization and norm_stats strategy"
git push
```

---

## Part 6：Phase D — 2×A800 从 base 全量微调（Week 2，主力工作）

**核心阶段**：用 2×A800 全量微调 `pi05_base → LIBERO`，30k step。A800 单卡 80GB 已经能放下 pi0.5，2 卡将 global batch 从 32 提到 64。本次真实墙钟时间约 28.4 小时（含最终 checkpoint 保存），后续排期应以实测而不是早期 16-20 小时估计为准。

### 6.1 技术

详见 `RUNBOOK.md` §8。当前默认按 2 卡跑。

先选两张空卡，优先同 NUMA 组，例如 `(4,5)` 或 `(6,7)`：

```bash
export PERSONAL_DIR=/cfsdata/chenjinfeng/projects/openpi-libero-reproduction
export RUN_ID=gpu_check_$(date +%Y%m%d_%H%M)
export EXP_DIR=$PERSONAL_DIR/experiments/$RUN_ID
mkdir -p "$EXP_DIR/logs"

nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | sort -k2 -t',' -rn \
    | tee "$EXP_DIR/logs/gpu_status.log"
```

判定标准：两张卡最好都 `memory.free > 75000 MiB`。

手动训练命令：

```bash
cd /cfsdata/chenjinfeng/projects/openpi
tmux new -s train_pi05_libero_2gpu

# tmux 内：假设 GPU 4,5 空闲；如果空的是 6,7，就改成 CUDA_VISIBLE_DEVICES=6,7
export EXP_NAME=pi05_libero_2gpu_$(date +%Y%m%d_%H%M)
export PERSONAL_DIR=/cfsdata/chenjinfeng/projects/openpi-libero-reproduction
export EXP_DIR=$PERSONAL_DIR/experiments/$EXP_NAME
mkdir -p "$EXP_DIR/logs"

CUDA_VISIBLE_DEVICES=4,5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_libero \
    --exp-name=$EXP_NAME \
    --overwrite \
    --fsdp-devices 2 \
    --batch-size 64 \
    --num-train-steps 30000 \
    --save-interval 5000 \
    2>&1 | tee "$EXP_DIR/logs/02_train.log"

# Ctrl+B D 离开
```

如果训练中断，使用同一个 `EXP_NAME` 续训：

```bash
cd /cfsdata/chenjinfeng/projects/openpi
export PERSONAL_DIR=/cfsdata/chenjinfeng/projects/openpi-libero-reproduction
export EXP_DIR=$PERSONAL_DIR/experiments/$EXP_NAME
mkdir -p "$EXP_DIR/logs"

CUDA_VISIBLE_DEVICES=4,5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_libero \
    --exp-name=$EXP_NAME \
    --resume \
    --fsdp-devices 2 \
    2>&1 | tee -a "$EXP_DIR/logs/02_train_resume.log"
```

或者用一键脚本。这个脚本会自动选两张空卡，batch 默认 64：

```bash
cd /cfsdata/chenjinfeng/projects/openpi-libero-reproduction
export EXP_NAME=pi05_libero_2gpu_v1_$(date +%Y%m%d_%H%M)
mkdir -p "experiments/$EXP_NAME/logs"
./scripts/run_full_pipeline.sh v1 2 \
    2>&1 | tee "experiments/$EXP_NAME/logs/run_full_pipeline.log"
```

如果要手动指定 GPU：

```bash
cd /cfsdata/chenjinfeng/projects/openpi-libero-reproduction
export EXP_NAME=pi05_libero_2gpu_v1_manual_$(date +%Y%m%d_%H%M)
mkdir -p "experiments/$EXP_NAME/logs"
GPU_IDS=4,5 ./scripts/run_full_pipeline.sh v1 2 \
    2>&1 | tee "experiments/$EXP_NAME/logs/run_full_pipeline.log"
```

### 6.2 Git 动作（训练过程中并行做）

本次训练实测约 28.4 小时。训练期间可同步整理实验记录：

```bash
cd ~/projects/openpi-libero-reproduction
export EXP_DIR=${EXP_DIR:-/cfsdata/chenjinfeng/projects/openpi-libero-reproduction/experiments/manual_docs_$(date +%Y%m%d_%H%M)}
mkdir -p "$EXP_DIR/logs"

# 更新 docs/method.md 中的真实配置和墙钟时间，不写占位结果

git add docs/method.md 2>&1 | tee "$EXP_DIR/logs/git_add_method.log"
git commit -m "docs: stub method documentation" 2>&1 | tee "$EXP_DIR/logs/git_commit_method.log"

# WandB 截图保存
# 具体截图清单见 docs/wandb.md。最低要求：
# - docs/figures/wandb_run_overview_2gpu.png
# - docs/figures/train_loss_2gpu_30k.png
# - docs/figures/grad_norm_2gpu_30k.png
git add docs/figures/ 2>&1 | tee "$EXP_DIR/logs/git_add_figures.log"
git commit -m "docs: add training curve snapshot day 1" 2>&1 | tee "$EXP_DIR/logs/git_commit_figures.log"
git push 2>&1 | tee "$EXP_DIR/logs/git_push.log"
```

### 6.3 跑你的官方协议评测

```bash
cd /cfsdata/chenjinfeng/projects/openpi-libero-reproduction
export PERSONAL_DIR=/cfsdata/chenjinfeng/projects/openpi-libero-reproduction
export RUN_ID=post_train_eval_$(date +%Y%m%d_%H%M)
export EVAL_RESULTS=$PERSONAL_DIR/experiments/$RUN_ID
mkdir -p "$EVAL_RESULTS" "$EVAL_RESULTS/logs"

cd /cfsdata/chenjinfeng/projects/openpi/examples/libero

export OPENPI_DATA_HOME=/cfsdata/chenjinfeng/openpi_cache
export HF_HOME=/cfsdata/chenjinfeng/hf_cache
export HF_LEROBOT_HOME=/cfsdata/chenjinfeng/datasets
export TMPDIR=/cfsdata/chenjinfeng/tmp
export MUJOCO_GL=egl
export PYTHONPATH=/cfsdata/chenjinfeng/projects/LIBERO:$PYTHONPATH

for SUITE in libero_spatial libero_object libero_goal libero_10
do
    echo "=== Evaluating $SUITE on your checkpoint ==="
    uv run main.py \
        --args.task-suite-name "$SUITE" \
        --args.host=localhost \
        --args.port=8001 \
        --args.video-out-path "$EVAL_RESULTS/videos/$SUITE" \
        2>&1 | tee "$EVAL_RESULTS/logs/eval_${SUITE}.log"
done

echo "suite,total_success_rate,total_episodes" > "$EVAL_RESULTS/sr_summary.csv"
for SUITE in libero_spatial libero_object libero_goal libero_10
do
    LOG="$EVAL_RESULTS/logs/eval_${SUITE}.log"
    SR=$(grep "Total success rate:" "$LOG" | tail -1 | awk '{print $NF}')
    EP=$(grep "Total episodes:" "$LOG" | tail -1 | awk '{print $NF}')
    echo "$SUITE,$SR,$EP" >> "$EVAL_RESULTS/sr_summary.csv"
done

cat "$EVAL_RESULTS/sr_summary.csv" | tee "$EVAL_RESULTS/logs/sr_summary.log"
```

### 6.4 归档你的评测结果

本次 2×A800 full fine-tune 已完成一组官方协议评测：

- Training run: `pi05_libero_2gpu_20260610_1529`
- Checkpoint: `/cfsdata/chenjinfeng/projects/openpi/checkpoints/pi05_libero/pi05_libero_2gpu_20260610_1529/29999`
- Eval run: `/cfsdata/chenjinfeng/projects/openpi-libero-reproduction/experiments/post_train_eval_20260612_1120`
- Protocol: 50 episodes/task × 10 tasks/suite = 500 episodes/suite

| Suite | Success rate | Episodes |
|---|---:|---:|
| libero_spatial | 0.984 | 500 |
| libero_object | 0.984 | 500 |
| libero_goal | 0.968 | 500 |
| libero_10 | 0.918 | 500 |
| **Average** | **0.9635** | **2000** |

`EGL_NOT_INITIALIZED` appeared after `Total success rate` / `Total episodes` were printed, during robosuite EGL cleanup. Treat this as a cleanup warning, not an invalid evaluation. `WARNING: Nan, Inf or huge value in QACC` appeared once at shutdown in the final log; the suite still completed all 500 episodes and printed the final rate.

```bash
cd /cfsdata/chenjinfeng/projects/openpi-libero-reproduction
export PERSONAL_DIR=/cfsdata/chenjinfeng/projects/openpi-libero-reproduction
export EXP_DIR=${EXP_DIR:-$PERSONAL_DIR/experiments/post_train_$(date +%Y%m%d_%H%M)}
mkdir -p "$EXP_DIR/logs"

export EXP_NAME=pi05_libero_2gpu_20260610_1529
export EVAL_RUN_ID=post_train_eval_20260612_1120
export EVAL_RESULTS=$PERSONAL_DIR/experiments/$EVAL_RUN_ID

if [ ! -f "$EVAL_RESULTS/sr_summary.csv" ]; then
    echo "Missing $EVAL_RESULTS/sr_summary.csv"
    echo "Please run 6.3 first."
    exit 1
fi

cp "$EVAL_RESULTS/sr_summary.csv" \
   "$PERSONAL_DIR/experiments/${EXP_NAME}_eval_sr_summary.csv" \
   2>&1 | tee "$EXP_DIR/logs/archive_sr_summary.log"

cat "$PERSONAL_DIR/experiments/${EXP_NAME}_eval_sr_summary.csv" \
    >> "$PERSONAL_DIR/experiments/results.csv" \
    2>> "$EXP_DIR/logs/append_results.log"

git add README.md "$PERSONAL_DIR/experiments/results.csv" \
    "$PERSONAL_DIR/experiments/${EXP_NAME}_eval_sr_summary.csv" \
    2>&1 | tee "$EXP_DIR/logs/git_add_results.log"
git commit -m "exp: 2xA800 full FT 30k step, avg SR=96.35%" 2>&1 | tee "$EXP_DIR/logs/git_commit_results.log"
git push 2>&1 | tee "$EXP_DIR/logs/git_push_results.log"
```

### 6.5 官方 baseline 归档（可选）

如果你只是想把**官方 baseline**复制一份留档，单独用下面这条：

```bash
cp /cfsdata/chenjinfeng/projects/openpi-libero-reproduction/experiments/baseline_official_50ep_20260607_1246/sr_summary.csv \
   /cfsdata/chenjinfeng/projects/openpi-libero-reproduction/experiments/baseline_official_50ep_20260607_1246/sr_summary_copy.csv
```

### 6.6 发 Release v0.1

```bash
git tag -a v0.1.0-base-reproduction -m "π0.5 base reproduction complete on 2×A800"
git push origin v0.1.0-base-reproduction
```

GitHub 网页 Releases → Draft → 选 tag → 写 release notes：

```markdown
## v0.1.0 — π0.5 Base Reproduction ✅

First milestone: full reproduction of π0.5 from pre-trained base to LIBERO
benchmark on 2×A800 in a shared cluster.

### Results
| Suite | Official ckpt (my reprod.) | My fine-tune | PI report |
|---|:-:|:-:|:-:|
| Spatial | 98.2% | 98.4% | 98.8% |
| Object | 98.8% | 98.4% | 98.2% |
| Goal | 96.8% | 96.8% | 98.0% |
| Long-10 | 92.6% | 91.8% | 92.4% |

### Highlights
- ✅ 2×A800 full fine-tuning with FSDP (no LoRA)
- ✅ Solved norm_stats compatibility issue
- ✅ Solved offline weight download via HTTPS
- Total training: ~28.4 hours including final checkpoint save

### Resources
- 📊 wandb: https://wandb.ai/3267189544-uestc/openpi/runs/hwdxnlvn
- 📝 blog: not published yet
- 📁 detailed report: [docs/baseline.md](docs/baseline.md)
```

### 6.7 第二篇博客

主题：**"openpi 微调踩坑实录：从 norm_stats 到外网受限下的权重分发"**

工程类博客高流量，对求职帮助大。

### 6.8 Phase D 验收

- [x] 一组 2×A800 30k step 训练完成
- [x] 微调 checkpoint 完成 50ep/task 正式评测
- [x] README 主表格更新 2×A800 微调结果
- [x] `v0.1.0-base-reproduction` tag 已推送
- [ ] GitHub Release 页面发布（tag 已存在）
- [ ] 第二篇博客发布

---

## Part 7：Phase E — 微调结果评测可视化（Week 3）

### 7.1 技术

详见 `RUNBOOK.md` §9-§10。50 episode 正式评测 + 三方对比图 + demo 视频。

WandB 截图要求见 `docs/wandb.md`。训练过程用 WandB 截图证明，最终 SR 用 eval log / `sr_summary.csv` 证明。

### 7.2 Git 动作

```bash
cd ~/projects/openpi-libero-reproduction
export EXP_DIR=${EXP_DIR:-/cfsdata/chenjinfeng/projects/openpi-libero-reproduction/experiments/visualize_$(date +%Y%m%d_%H%M)}
mkdir -p "$EXP_DIR/logs"

# 跑可视化脚本（使用已归档的真实结果）
uv run python scripts/plot_results.py 2>&1 | tee "$EXP_DIR/logs/plot_results.log"

# 挑 demo 视频
mkdir -p docs/demos
cp /cfsdata/chenjinfeng/datasets/eval_videos/$EXP_NAME/libero_10/task_0_seed_0.mp4 \
   docs/demos/long_success.mp4 \
   2>&1 | tee "$EXP_DIR/logs/copy_demo.log"
# ≤50MB 的进 repo，>50MB 的上 HuggingFace

# Hero GIF for README
ffmpeg -i docs/demos/long_success.mp4 \
    -vf "fps=10,scale=480:-1:flags=lanczos" \
    -loop 0 docs/demo.gif \
    2>&1 | tee "$EXP_DIR/logs/ffmpeg_demo_gif.log"

git add docs/figures/ docs/demos/ docs/demo.gif README.md 2>&1 | tee "$EXP_DIR/logs/git_add_visuals.log"
git commit -m "docs: add 3-way SR comparison, demo videos, hero GIF" 2>&1 | tee "$EXP_DIR/logs/git_commit_visuals.log"
git push 2>&1 | tee "$EXP_DIR/logs/git_push_visuals.log"
```

### 7.3 第三篇博客

主题：**"2×A800 跑通 π0.5 LIBERO 全量微调：从 baseline 到自己的 checkpoint"**

综合性总结博客，可以做简历附件链接。

---

## Part 8：Phase F — DGTE 推理创新与配对评测

本阶段选择训练免费的 DGTE（Disagreement-Gated Temporal Ensemble）作为第一个可验证创新。原因是它直接利用 pi0.5 已输出但被客户端丢弃的重叠 action chunk，不改 checkpoint，不引入尚未具备的深度/点云数据，也能做严格的同模型 A/B。

### 8.1 已实现

- `src/openpi_libero_reproduction/temporal_ensemble.py`：时序对齐、freshness weighting、分歧门控、夹爪离散保护。
- `scripts/eval_libero_temporal.py`：同一 LIBERO client 支持 `baseline` / `dgte`。
- `scripts/run_temporal_ablation.sh`：同一个 server 和 checkpoint 下依次跑两种 controller。
- `tests/test_temporal_ensemble.py`：CPU 单元测试。

### 8.2 评测命令

```bash
cd /cfsdata/chenjinfeng/projects/openpi-libero-reproduction

EXP_NAME=pi05_libero_2gpu_20260610_1529 \
GPU_ID=<空闲卡> \
N_EPISODES_PER_TASK=10 \
./scripts/run_temporal_ablation.sh
```

10 episode/task 只用于 smoke test。正式表格必须用 `N_EPISODES_PER_TASK=50`，并同时跑完四个 suite。GPU 忙时保持结果为 `pending`，不能用假设提升填表。

### 8.3 后续多模态方向

点云/深度/雷达融合保留为更长期研究方向，但必须在数据、标定和计算资源就绪后单独立项。当前仓库不宣称尚未实现的多模态结果。

---

## Part 9：里程碑发布策略

| 里程碑 | Tag | 何时发 | 内容 |
|---|---|---|---|
| `v0.1.0-base-reproduction` | Phase D 末 | 2×A800 full FT 跑出 SR | 第一个里程碑 |
| `v0.2.0-dgte` | Phase F 末 | DGTE 50 episode 配对评测完成 | 第一组创新数字 |
| `v0.3.0-analysis` | DGTE 后 | 参数消融 + failure analysis | 报告级分析 |
| `v1.0.0` | 后续研究完成 | 经验证的完整方法与复现实验 | 稳定版本 |

Release notes 模板：

```markdown
## [版本] — [一句话定位]

[一段总结]

### Highlights
- 🎯 [关键成果]
- ⚡ [关键成果]

### Results
[表格]

### Resources
- 📊 wandb: [link]
- 📝 blog: [link]
- 📁 docs: [docs/xxx.md]
```

---

## Part 10：持续输出体系

### 10.1 博客节奏

| Phase | 主题 | 平台 |
|---|---|---|
| A 后 | "A800 集群上 openpi 环境踩坑实录" | 知乎 |
| B 后 | "手把手跑通 π0.5 LIBERO 评测" | 知乎 + CSDN |
| D 后 | "openpi 微调踩坑：norm_stats 怎么干掉 95% SR" | 知乎（高流量） |
| E 后 | "在 1×A800 上跑出 SR 90+：π0.5 LIBERO 全记录" | 全平台 |
| F 后 | "我为什么选 [方向] 做多模态融合：决策过程" | 知乎 |
| 多模态完 | "方法、实验、踩坑" | 知乎 + Twitter |

每篇发完链接进 README "Related blogs" 和 Profile README。

### 10.2 WandB 公开

WandB → 项目 → Settings → Privacy → **Public**。把 URL 进 README 和每次 release。

### 10.3 GitHub 优化

- Topics：`embodied-ai`、`vla`、`pi0`、`libero`、`reproduction`、`paligemma`
- Social preview：上传一张 1280×640 图（demo + 项目名 + 你名字）
- 自己 star 自己的 repo（正常操作）

### 10.4 让别人发现你

- Star / Watch 相关仓库：openpi、lerobot、SpatialVLA、PointVLA、Galaxea-VLA
- 在他们 Issues 高质量回答 → 你的 profile 出现在 issue 页
- Twitter follow Sergey Levine、Chelsea Finn、Karpathy、王鹤等

---

## Part 11：简历整合

### 11.1 简历第一行

```
[你的中文名]    📧 your.email@xxx.edu.cn    📱 [手机]
GitHub: github.com/Jinfeng50    Blog / 知乎: 按实际主页补充
```

### 11.2 项目段落末尾加

```
📂 github.com/Jinfeng50/openpi-libero-reproduction（含复现代码、训练曲线、SR 结果和 DGTE A/B 评测入口）
```

### 11.3 HR 视角检查清单

让一个不懂技术的朋友打开你 GitHub，30 秒后问他：

- [ ] 5 秒内能看出你在做什么？（pinned + 描述）
- [ ] 有"持续工作"的感觉？（绿格子图、commit history）
- [ ] 有"成果"的感觉？（Releases、视频、SR 数字）
- [ ] 能找到联系方式？（profile bio）

任何一条 No 都回头补。

### 11.4 面试 GitHub 标准动作

面试官问"讲一个项目"时：

1. 打开 `openpi-libero-reproduction`
2. 滚到 Results 表 → **讲数字**
3. 打开 demo GIF / 视频 → **视觉冲击**
4. 打开 Releases → **讲里程碑**
5. 打开 `docs/method.md` → **讲技术**
6. 打开 `docs/baseline.md` → **讲严谨**
7. 链接到 wandb → **讲实验充分**

5 分钟讲完，**95% 面试官被打动**。

---

## 附录 A：故障排查

### A.1 `git push` 卡住或 timeout
→ RUNBOOK §11 + 本文 §1.3 的 443 配置。

### A.2 误 commit 大文件
```bash
pip install git-filter-repo
git filter-repo --path 大文件路径 --invert-paths
git push --force
```
⚠️ force push 改写历史。

### A.3 误 commit 密钥
1. 立刻去对应平台 revoke
2. 用 filter-repo 清掉
3. 重新生成

### A.4 submodule 没拉取
```bash
git submodule update --init --recursive
```

### A.5 远程仓库需要鉴权
检查 `git remote -v`，URL 应是 `git@github.com:xxx`，不是 `https://`。

---

## 附录 B：每日 / 每周 / 每月 checklist

### 每天（30 分钟内的 git 时间）
- [ ] 早 `git pull`
- [ ] 工作中 ≥1 个 commit
- [ ] 晚 LEARNING_LOG entry
- [ ] 实验有数字 → 加进 docs/

### 每周日晚（30 分钟）
- [ ] LEARNING_LOG 写周回顾
- [ ] README 主表格是否需要更新
- [ ] 本周 commit message 质量自查
- [ ] 计划下周 3 个主任务

### 每月最后一天（1 小时）
- [ ] 月度里程碑 check
- [ ] LEARNING_LOG 写月回顾
- [ ] 技能矩阵打分
- [ ] 看是否到 release 节点
- [ ] 写一篇博客（如果适合）

---

## 附录 C：速查命令表

```bash
# 日常
git status / git diff / git add . / git commit -m "..." / git push
git log --oneline -10
git log --graph --oneline --all

# 分支
git checkout -b feat/xxx
git checkout main && git pull

# Tag & Release
git tag -a v0.1.0 -m "..."
git push origin v0.1.0

# 撤销
git restore <文件>            # 撤销未暂存
git restore --staged <文件>   # 撤销 add
git reset --soft HEAD~1      # 撤上次 commit 保留改动
git reset --hard HEAD~1      # 撤上次 commit 丢改动（危险）

# 远程
git remote -v
git remote set-url origin <new-url>

# Submodule
git submodule update --init --recursive
```

---

## 结语

把这份指南进 repo：

```bash
cp FULL_GUIDE.md ~/projects/openpi-libero-reproduction/
cd ~/projects/openpi-libero-reproduction
git add FULL_GUIDE.md
git commit -m "docs: add full project guide (A800 single-GPU version)"
git push
```

**核心心法**：

1. **80GB A800 全量微调，不再需要 LoRA**——LoRA 是 4090 时代的妥协，你不需要。
2. **共享集群常态是 1-2 卡可用**，所有训练流程围绕这个现实设计。
3. **开源是过程，不是结果**——6 月空仓库 + 12 月完整作品 = 可信成长曲线。

跑命令遇任何报错就贴给我，咱们继续。
