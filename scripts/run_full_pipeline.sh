#!/bin/bash
# ==============================================================================
# π0 / π0.5 LIBERO 全链路一键脚本
# ==============================================================================
# 串联：环境检查 → norm_stats → 训练 → 启 server → 评测 → 汇总结果
#
# 用法：
#   ./run_full_pipeline.sh                              # 用默认配置
#   ./run_full_pipeline.sh pi0_libero_low_mem_finetune  # 指定 config
#   ./run_full_pipeline.sh pi05_libero v2 8             # config + suffix + GPU数
#
# 跳过某些阶段（环境变量控制）：
#   SKIP_NORM=1 ./run_full_pipeline.sh
#   SKIP_TRAIN=1 ./run_full_pipeline.sh    # 只跑评测（要有现成 ckpt）
#   SKIP_EVAL=1  ./run_full_pipeline.sh    # 只训练不评测
#
# 作者：[你的名字]   最后修改：2026-XX-XX
# ==============================================================================

set -e
set -o pipefail

# ---------- 用户可调参数 ----------
CONFIG=${1:-pi0_libero_low_mem_finetune}            # 训练 config
EXP_SUFFIX=${2:-v1}                                 # 实验后缀
NUM_GPUS=${3:-1}                                    # 用几张 GPU

NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-30000}
BATCH_SIZE=${BATCH_SIZE:-32}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
N_EPISODES_PER_TASK=${N_EPISODES_PER_TASK:-50}
EVAL_CKPT_STEP=${EVAL_CKPT_STEP:-$NUM_TRAIN_STEPS}  # 评测哪一步的 ckpt
SERVER_PORT=${SERVER_PORT:-8000}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}

# ---------- 路径 ----------
OPENPI_DIR=/cfsdata/chenjinfeng/projects/openpi
EVAL_VIDEO_ROOT=/cfsdata/chenjinfeng/datasets/eval_videos
EXP_NAME="${CONFIG}_${EXP_SUFFIX}_$(date +%Y%m%d_%H%M%S)"
EXP_DIR="$OPENPI_DIR/experiments/$EXP_NAME"
CKPT_DIR="$OPENPI_DIR/checkpoints/$EXP_NAME"

# ---------- 环境变量 ----------
export OPENPI_DATA_HOME=/cfsdata/chenjinfeng/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl
export GIT_LFS_SKIP_SMUDGE=1
export WANDB_PROJECT=${WANDB_PROJECT:-openpi-libero}
export WANDB_NAME=$EXP_NAME
export HF_LEROBOT_HOME=/cfsdata/chenjinfeng/datasets

# 多卡 GPU 字符串
if [ "$NUM_GPUS" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
else
    GPU_IDS=$(seq -s, 0 $((NUM_GPUS-1)))
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_IDS}
fi

# ---------- 美化输出 ----------
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

log()      { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
log_ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"; }
log_warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠${NC} $*"; }
log_err()  { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*"; }
hr() { echo -e "${BLUE}=========================================================${NC}"; }

# ---------- 错误处理 ----------
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
        log "Cleaning up policy server PID $SERVER_PID..."
        kill $SERVER_PID 2>/dev/null || true
        sleep 2
        kill -9 $SERVER_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ==============================================================================
# Stage 0: 准备 & 健康检查
# ==============================================================================
hr
log "Experiment: $EXP_NAME"
log "Config:     $CONFIG"
log "GPUs:       $CUDA_VISIBLE_DEVICES ($NUM_GPUS)"
log "Steps:      $NUM_TRAIN_STEPS"
log "Batch:      $BATCH_SIZE"
log "Save dir:   $CKPT_DIR"
log "Exp dir:    $EXP_DIR"
hr

mkdir -p $EXP_DIR/logs $EXP_DIR/eval_logs
cd $OPENPI_DIR

log "Sanity check..."
uv run python -c "import openpi" || { log_err "openpi not importable"; exit 1; }
uv run python -c "import jax; assert len(jax.devices()) >= 1" || { log_err "no JAX device"; exit 1; }
log_ok "Environment OK"

# 验证 base 权重存在
uv run python -c "
from openpi.training import config as C
cfg = C.get_config('$CONFIG')
print('Weight loader:', cfg.weight_loader)
" 2>&1 | tee $EXP_DIR/logs/00_config_info.log

# ==============================================================================
# Stage 1: compute_norm_stats
# ==============================================================================
hr
if [ "${SKIP_NORM:-0}" = "1" ]; then
    log_warn "Skipping norm_stats (SKIP_NORM=1)"
else
    log "Stage 1/4: compute_norm_stats for $CONFIG"
    if [ -f "assets/$CONFIG/norm_stats.json" ]; then
        log_warn "norm_stats already exists at assets/$CONFIG/norm_stats.json — skipping"
    else
        uv run scripts/compute_norm_stats.py --config-name $CONFIG \
            2>&1 | tee $EXP_DIR/logs/01_norm_stats.log
        log_ok "norm_stats computed"
    fi
fi

# ==============================================================================
# Stage 2: 训练
# ==============================================================================
hr
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    log_warn "Skipping training (SKIP_TRAIN=1)"
else
    log "Stage 2/4: training ($NUM_TRAIN_STEPS steps, batch=$BATCH_SIZE)"
    log "Wandb run name: $WANDB_NAME"
    log "Logs streaming to: $EXP_DIR/logs/02_train.log"

    if [ "$NUM_GPUS" -gt 1 ]; then
        FSDP_FLAG="--fsdp-devices $NUM_GPUS"
    else
        FSDP_FLAG=""
    fi

    TRAIN_START=$(date +%s)
    uv run scripts/train.py $CONFIG \
        --exp-name=$EXP_NAME \
        --overwrite \
        --num-train-steps $NUM_TRAIN_STEPS \
        --batch-size $BATCH_SIZE \
        --save-interval $SAVE_INTERVAL \
        $FSDP_FLAG \
        2>&1 | tee $EXP_DIR/logs/02_train.log
    TRAIN_END=$(date +%s)
    TRAIN_HOURS=$(echo "scale=2; ($TRAIN_END - $TRAIN_START) / 3600" | bc)
    log_ok "Training done in ${TRAIN_HOURS} hours"
fi

# 检查 ckpt 存在
EVAL_CKPT="$CKPT_DIR/$EVAL_CKPT_STEP"
if [ ! -d "$EVAL_CKPT" ]; then
    log_err "Checkpoint not found: $EVAL_CKPT"
    log_err "Available: $(ls $CKPT_DIR 2>/dev/null | tr '\n' ' ')"
    exit 1
fi

# ==============================================================================
# Stage 3: 启动 policy server（后台）
# ==============================================================================
hr
if [ "${SKIP_EVAL:-0}" = "1" ]; then
    log_warn "Skipping eval (SKIP_EVAL=1). Pipeline done."
    exit 0
fi

log "Stage 3/4: starting policy server"
log "Checkpoint: $EVAL_CKPT"
log "Port: $SERVER_PORT"

# 评测只用 GPU 0
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=$CONFIG \
    --policy.dir=$EVAL_CKPT \
    --port=$SERVER_PORT \
    > $EXP_DIR/logs/03_server.log 2>&1 &
SERVER_PID=$!
log "Server started, PID=$SERVER_PID"

# 等 server 起来（探测端口）
log "Waiting for server to be ready..."
for i in {1..60}; do
    if nc -z localhost $SERVER_PORT 2>/dev/null; then
        log_ok "Server is up after ${i}s"
        break
    fi
    if [ $i -eq 60 ]; then
        log_err "Server didn't come up in 60s. Check $EXP_DIR/logs/03_server.log"
        exit 1
    fi
    sleep 1
done

# 多等 30 秒，让模型完全加载
log "Waiting 30s for model warmup..."
sleep 30

# ==============================================================================
# Stage 4: LIBERO 评测
# ==============================================================================
hr
log "Stage 4/4: LIBERO evaluation"
log "Suites: $SUITES"
log "Episodes per task: $N_EPISODES_PER_TASK"

cd $OPENPI_DIR/examples/libero
EVAL_START=$(date +%s)

# 汇总 csv 表头
SUMMARY_CSV=$EXP_DIR/sr_summary.csv
echo "date,exp_name,suite,n_episodes,success_rate" > $SUMMARY_CSV

for SUITE in $SUITES; do
    log "Evaluating $SUITE..."
    VIDEO_OUT=$EVAL_VIDEO_ROOT/$EXP_NAME/$SUITE
    mkdir -p $VIDEO_OUT

    uv run main.py \
        --args.task-suite-name $SUITE \
        --args.num-trials-per-task $N_EPISODES_PER_TASK \
        --args.video-out-path $VIDEO_OUT \
        --args.host=localhost \
        --args.port=$SERVER_PORT \
        2>&1 | tee $EXP_DIR/eval_logs/eval_${SUITE}.log || true

    # 从日志 grep SR
    SR=$(grep -oP "Success rate: \K[0-9.]+" $EXP_DIR/eval_logs/eval_${SUITE}.log | tail -1)
    if [ -z "$SR" ]; then
        SR="N/A"
        log_warn "Cannot parse SR for $SUITE"
    else
        log_ok "$SUITE: SR = $SR"
    fi
    echo "$(date +%Y-%m-%d),$EXP_NAME,$SUITE,$N_EPISODES_PER_TASK,$SR" >> $SUMMARY_CSV
done

EVAL_END=$(date +%s)
EVAL_HOURS=$(echo "scale=2; ($EVAL_END - $EVAL_START) / 3600" | bc)

# ==============================================================================
# 完成 & 汇总
# ==============================================================================
hr
log_ok "Pipeline complete!"
log "Total eval time: ${EVAL_HOURS} hours"
echo ""
log "Summary CSV: $SUMMARY_CSV"
cat $SUMMARY_CSV
echo ""
log "Demo videos: $EVAL_VIDEO_ROOT/$EXP_NAME/"
log "All logs:    $EXP_DIR/"
log "Checkpoint:  $EVAL_CKPT"
echo ""
log "Next steps:"
echo "  1. 把 $SUMMARY_CSV 内容追加到 experiments/results.csv"
echo "  2. 挑 3 段 demo 视频拷到 experiments/demo_videos/"
echo "  3. 在 LEARNING_LOG.md 里写今天的 entry"
echo "  4. wandb 截图保存到 experiments/figures/"
hr
