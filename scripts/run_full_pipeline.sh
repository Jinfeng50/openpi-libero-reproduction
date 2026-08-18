#!/bin/bash
# ==============================================================================
# π0.5 LIBERO 全链路一键脚本（A800 版）
# ==============================================================================
# 串联：环境检查 → norm_stats → 训练 → 启 server → 评测 → 汇总
#
# 用法：
#   ./run_full_pipeline.sh                       # 默认：1 卡，自动选空闲卡
#   ./run_full_pipeline.sh v2                    # 自定义实验后缀
#   ./run_full_pipeline.sh v2 2                  # 2 卡
#   ./run_full_pipeline.sh v3 4                  # 4 卡
#   GPU_IDS=5,6 ./run_full_pipeline.sh v4        # 手动指定 GPU
#
# 跳过阶段：
#   SKIP_NORM=1   ./run_full_pipeline.sh        # 跳过 norm_stats
#   EXP_NAME=<已有实验> SKIP_TRAIN=1 ./run_full_pipeline.sh  # 只评测
#   SKIP_EVAL=1   ./run_full_pipeline.sh        # 只训不评
#
# 作者：chenjinfeng    最后更新：2026-08
# ==============================================================================

set -e
set -o pipefail

# ---------- 用户可调参数 ----------
EXP_SUFFIX=${1:-v1}
NUM_GPUS=${2:-1}

CONFIG=${CONFIG:-pi05_libero}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-30000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
N_EPISODES_PER_TASK=${N_EPISODES_PER_TASK:-50}
# openpi writes the final 30k-step checkpoint as 29999.  Leave this empty so
# the script can resolve either the requested step or the latest available one.
EVAL_CKPT_STEP=${EVAL_CKPT_STEP:-}
SERVER_PORT=${SERVER_PORT:-8000}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}

# 根据 GPU 数自动定 batch_size
case $NUM_GPUS in
    1) BATCH_SIZE=${BATCH_SIZE:-32} ;;
    2) BATCH_SIZE=${BATCH_SIZE:-64} ;;
    4) BATCH_SIZE=${BATCH_SIZE:-128} ;;
    8) BATCH_SIZE=${BATCH_SIZE:-256} ;;
    *) BATCH_SIZE=${BATCH_SIZE:-32} ;;
esac

# ---------- 路径 ----------
OPENPI_DIR=/cfsdata/chenjinfeng/projects/openpi
PERSONAL_DIR=/cfsdata/chenjinfeng/projects/openpi-libero-reproduction
EVAL_VIDEO_ROOT=/cfsdata/chenjinfeng/datasets/eval_videos
EXP_NAME=${EXP_NAME:-"${CONFIG}_${NUM_GPUS}gpu_${EXP_SUFFIX}_$(date +%Y%m%d_%H%M)"}
EXP_DIR="$PERSONAL_DIR/experiments/$EXP_NAME"
CKPT_DIR="$OPENPI_DIR/checkpoints/$CONFIG/$EXP_NAME"

# ---------- 环境变量 ----------
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/cfsdata/chenjinfeng/openpi_cache}
export HF_HOME=${HF_HOME:-/cfsdata/chenjinfeng/hf_cache}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/cfsdata/chenjinfeng/datasets}
export TMPDIR=${TMPDIR:-/cfsdata/chenjinfeng/tmp}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export MUJOCO_GL=egl
export GIT_LFS_SKIP_SMUDGE=1
export WANDB_PROJECT=${WANDB_PROJECT:-openpi-libero}
export WANDB_NAME=$EXP_NAME

# ---------- 美化输出 ----------
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
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
# 自动选空闲 GPU
# ==============================================================================
auto_pick_gpus() {
    local n=$1
    timeout 15 nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        sort -k2 -t',' -rn | \
        awk -F', ' -v n=$n '
            NR<=n {
                if ($2+0 > 75000) { gpus[NR]=$1; count++ }
                else { print "INSUFFICIENT" > "/dev/stderr"; exit 1 }
            }
            END {
                if (count < n) exit 1
                out=""
                for (i=1; i<=n; i++) out = out (i==1 ? "" : ",") gpus[i]
                print out
            }'
}

if [ -z "$GPU_IDS" ]; then
    log "Auto-picking $NUM_GPUS free GPU(s)..."
    GPU_IDS=$(auto_pick_gpus $NUM_GPUS) || {
        log_err "Need $NUM_GPUS GPUs with >75GB free, but not enough available."
        log_err "Current GPU status:"
        timeout 15 nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
            sort -k2 -t',' -rn | awk -F', ' '{ printf "  GPU %s: %5d MiB free\n", $1, $2 }'
        exit 1
    }
fi
export CUDA_VISIBLE_DEVICES=$GPU_IDS

GPU_COUNT=$(awk -F',' '{print NF}' <<< "$GPU_IDS")
if [ "$GPU_COUNT" -ne "$NUM_GPUS" ]; then
    log_err "NUM_GPUS=$NUM_GPUS but GPU_IDS=$GPU_IDS contains $GPU_COUNT device(s)"
    exit 1
fi

# ==============================================================================
# Stage 0: 准备 & 健康检查
# ==============================================================================
hr
log "Experiment: $EXP_NAME"
log "Config:     $CONFIG"
log "GPUs:       $CUDA_VISIBLE_DEVICES ($NUM_GPUS)"
log "Steps:      $NUM_TRAIN_STEPS"
log "Batch:      $BATCH_SIZE"
log "Ckpt:       $CKPT_DIR"
log "Exp logs:   $EXP_DIR"
hr

mkdir -p "$EXP_DIR/logs" "$EXP_DIR/eval_logs" "$TMPDIR"
cd $OPENPI_DIR

log "Sanity check..."
uv run python -c "
import torch, jax, transformers
assert transformers.__version__ == '4.53.2', 'transformers must be 4.53.2'
print(f'torch: {torch.__version__}, devices={torch.cuda.device_count()}')
print(f'jax:   {jax.__version__}, devices={len(jax.devices())}')
" 2>&1 | tee $EXP_DIR/logs/00_env_check.log || {
    log_err "Env check failed"
    exit 1
}
log_ok "Environment OK"

# 看一下 config 配置
uv run python -c "
from openpi.training import config as C
cfg = C.get_config('$CONFIG')
print('Weight loader:', cfg.weight_loader)
" 2>&1 | tee -a $EXP_DIR/logs/00_env_check.log

# ==============================================================================
# Stage 1: compute_norm_stats
# ==============================================================================
hr
if [ "${SKIP_NORM:-0}" = "1" ]; then
    log_warn "Skipping norm_stats (SKIP_NORM=1)"
elif find "assets/$CONFIG" -name norm_stats.json -type f -print -quit | grep -q .; then
    log_warn "norm_stats exists under assets/$CONFIG — skipping"
else
    log "Stage 1/4: compute_norm_stats for $CONFIG"
    uv run scripts/compute_norm_stats.py --config-name $CONFIG \
        2>&1 | tee $EXP_DIR/logs/01_norm_stats.log
    log_ok "norm_stats computed"
fi

# ==============================================================================
# Stage 2: 训练
# ==============================================================================
hr
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    log_warn "Skipping training (SKIP_TRAIN=1)"
else
    log "Stage 2/4: training ($NUM_TRAIN_STEPS steps, batch=$BATCH_SIZE, GPUs=$CUDA_VISIBLE_DEVICES)"
    log "Logs: $EXP_DIR/logs/02_train.log"
    log "Wandb name: $WANDB_NAME"

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
    TRAIN_HOURS=$(awk -v elapsed="$((TRAIN_END - TRAIN_START))" 'BEGIN { printf "%.2f", elapsed / 3600 }')
    log_ok "Training done in ${TRAIN_HOURS} hours"
fi

# 检查 ckpt 存在。训练结束后通常是 NUM_TRAIN_STEPS-1，兼容两种命名。
if [ -n "$EVAL_CKPT_STEP" ]; then
    EVAL_CKPT="$CKPT_DIR/$EVAL_CKPT_STEP"
else
    EVAL_CKPT="$CKPT_DIR/$NUM_TRAIN_STEPS"
    if [ ! -d "$EVAL_CKPT" ] && [ "$NUM_TRAIN_STEPS" -gt 0 ]; then
        EVAL_CKPT="$CKPT_DIR/$((NUM_TRAIN_STEPS - 1))"
    fi
fi
if [ ! -d "$EVAL_CKPT" ]; then
    log_err "Checkpoint not found: $EVAL_CKPT"
    log_err "Available: $(ls "$CKPT_DIR" 2>/dev/null | tr '\n' ' ')"
    [ "${SKIP_EVAL:-0}" = "1" ] && exit 0 || exit 1
fi

# ==============================================================================
# Stage 3: 启动 policy server
# ==============================================================================
hr
if [ "${SKIP_EVAL:-0}" = "1" ]; then
    log_warn "Skipping eval (SKIP_EVAL=1). Done."
    exit 0
fi

log "Stage 3/4: starting policy server"
log "Checkpoint: $EVAL_CKPT"
log "Port:       $SERVER_PORT"

# 评测只用 1 张卡（GPU_IDS 的第一张）
EVAL_GPU=$(echo $GPU_IDS | cut -d',' -f1)
log "Server will use GPU $EVAL_GPU"

CUDA_VISIBLE_DEVICES=$EVAL_GPU uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=$CONFIG \
    --policy.dir=$EVAL_CKPT \
    --port=$SERVER_PORT \
    > $EXP_DIR/logs/03_server.log 2>&1 &
SERVER_PID=$!
log "Server started, PID=$SERVER_PID"

# 等 server 起来
log "Waiting for server (max 60s)..."
for i in {1..60}; do
    if (command -v nc >/dev/null && nc -z localhost "$SERVER_PORT" 2>/dev/null) || \
       (exec 3<>"/dev/tcp/127.0.0.1/$SERVER_PORT") 2>/dev/null; then
        log_ok "Server up after ${i}s"
        break
    fi
    if [ $i -eq 60 ]; then
        log_err "Server didn't come up in 60s. See $EXP_DIR/logs/03_server.log"
        exit 1
    fi
    sleep 1
done

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

SUMMARY_CSV=$EXP_DIR/sr_summary.csv
echo "date,exp_name,suite,episodes_per_task,total_episodes,success_rate" > "$SUMMARY_CSV"

for SUITE in $SUITES; do
    log "Evaluating $SUITE..."
    VIDEO_OUT="$EVAL_VIDEO_ROOT/$EXP_NAME/$SUITE"
    mkdir -p "$VIDEO_OUT"

    if ! uv run main.py \
        --args.task-suite-name $SUITE \
        --args.num-trials-per-task $N_EPISODES_PER_TASK \
        --args.video-out-path "$VIDEO_OUT" \
        --args.host=localhost \
        --args.port=$SERVER_PORT \
        2>&1 | tee "$EXP_DIR/eval_logs/eval_${SUITE}.log"; then
        log_err "Evaluation failed for $SUITE; see $EXP_DIR/eval_logs/eval_${SUITE}.log"
        exit 1
    fi

    SR=$(grep -oP "Total success rate: \K[0-9.]+" "$EXP_DIR/eval_logs/eval_${SUITE}.log" | tail -1)
    EPISODES=$(grep -oP "Total episodes: \K[0-9]+" "$EXP_DIR/eval_logs/eval_${SUITE}.log" | tail -1)
    if [ -z "$SR" ] || [ -z "$EPISODES" ]; then
        log_err "Cannot parse final SR/episode count for $SUITE"
        exit 1
    fi
    log_ok "$SUITE: SR = $SR ($EPISODES episodes)"
    echo "$(date +%Y-%m-%d),$EXP_NAME,$SUITE,$N_EPISODES_PER_TASK,$EPISODES,$SR" >> "$SUMMARY_CSV"
done

EVAL_END=$(date +%s)
EVAL_HOURS=$(awk -v elapsed="$((EVAL_END - EVAL_START))" 'BEGIN { printf "%.2f", elapsed / 3600 }')

# ==============================================================================
# 完成 & 汇总
# ==============================================================================
hr
log_ok "Pipeline complete!"
log "Eval time:   ${EVAL_HOURS} hours"
echo ""
log "Summary CSV: $SUMMARY_CSV"
cat "$SUMMARY_CSV"
echo ""
log "Demo videos: $EVAL_VIDEO_ROOT/$EXP_NAME/"
log "Logs:        $EXP_DIR/"
log "Checkpoint:  $EVAL_CKPT"
echo ""
log "Next steps:"
echo "  1. Review $SUMMARY_CSV and update the wide experiments/results.csv ledger."
echo "  2. Save selected WandB evidence under $PERSONAL_DIR/docs/figures/."
echo "  3. Update docs/results.md and README with verified numbers only."
hr
