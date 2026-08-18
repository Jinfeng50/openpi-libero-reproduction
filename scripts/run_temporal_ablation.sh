#!/usr/bin/env bash
# Run a strict baseline vs DGTE client-side ablation against one checkpoint.
set -euo pipefail

OPENPI_DIR=${OPENPI_DIR:-/cfsdata/chenjinfeng/projects/openpi}
PERSONAL_DIR=${PERSONAL_DIR:-/cfsdata/chenjinfeng/projects/openpi-libero-reproduction}
LIBERO_DIR=${LIBERO_DIR:-/cfsdata/chenjinfeng/projects/LIBERO}
CONFIG=${CONFIG:-pi05_libero}
EXP_NAME=${EXP_NAME:?Set EXP_NAME to the checkpoint experiment name}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-29999}
N_EPISODES_PER_TASK=${N_EPISODES_PER_TASK:-50}
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
SERVER_PORT=${SERVER_PORT:-8002}
GPU_ID=${GPU_ID:-}
RUN_ID=${RUN_ID:-dgte_ablation_$(date +%Y%m%d_%H%M)}
EXP_DIR="$PERSONAL_DIR/experiments/$RUN_ID"
CKPT="$OPENPI_DIR/checkpoints/$CONFIG/$EXP_NAME/$CHECKPOINT_STEP"

if [[ ! -d "$CKPT" ]]; then
    echo "Checkpoint does not exist: $CKPT" >&2
    exit 1
fi

if [[ -z "$GPU_ID" ]]; then
    if ! GPU_ID=$(timeout 15 nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -k2 -t',' -rn \
        | awk -F', ' '$2 + 0 > 75000 { print $1; exit }'); then
        echo "GPU query timed out or failed; set GPU_ID explicitly after checking the node." >&2
        exit 1
    fi
fi
if [[ -z "$GPU_ID" ]]; then
    echo "No GPU with more than 75 GiB free. Set GPU_ID explicitly after checking nvidia-smi." >&2
    exit 1
fi

export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/cfsdata/chenjinfeng/openpi_cache}
export HF_HOME=${HF_HOME:-/cfsdata/chenjinfeng/hf_cache}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/cfsdata/chenjinfeng/datasets}
export TMPDIR=${TMPDIR:-/cfsdata/chenjinfeng/tmp}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH="$LIBERO_DIR:$OPENPI_DIR/third_party/libero:$PERSONAL_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$EXP_DIR/logs" "$EXP_DIR/videos"
SUMMARY="$EXP_DIR/sr_summary.csv"
echo "date,run_id,checkpoint,controller,suite,episodes,success_rate" > "$SUMMARY"

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM
server_ready() {
    if command -v nc >/dev/null 2>&1; then
        nc -z -w 1 127.0.0.1 "$SERVER_PORT" >/dev/null 2>&1
    else
        grep -q "server listening on 0.0.0.0:$SERVER_PORT" "$EXP_DIR/logs/server.log"
    fi
}

cd "$OPENPI_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" uv run scripts/serve_policy.py \
    --port "$SERVER_PORT" \
    policy:checkpoint \
    --policy.config="$CONFIG" \
    --policy.dir="$CKPT" \
    > "$EXP_DIR/logs/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 120); do
    if server_ready; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Policy server exited; see $EXP_DIR/logs/server.log" >&2
        exit 1
    fi
    sleep 1
done

if ! server_ready; then
    echo "Policy server did not listen on port $SERVER_PORT" >&2
    exit 1
fi

for controller in baseline dgte; do
    for suite in $SUITES; do
        log="$EXP_DIR/logs/${controller}_${suite}.log"
        video_dir="$EXP_DIR/videos/${controller}/${suite}"
        mkdir -p "$video_dir"
        echo "Evaluating controller=$controller suite=$suite"
        uv run python "$PERSONAL_DIR/scripts/eval_libero_temporal.py" \
            --host localhost \
            --port "$SERVER_PORT" \
            --task-suite-name "$suite" \
            --num-trials-per-task "$N_EPISODES_PER_TASK" \
            --video-out-path "$video_dir" \
            --controller "$controller" \
            2>&1 | tee "$log"
        sr=$(awk '/Total success rate:/ { value=$NF } END { print value }' "$log")
        episodes=$(awk '/Total episodes:/ { value=$NF } END { print value }' "$log")
        [[ "$sr" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Could not parse SR from $log" >&2; exit 1; }
        [[ "$episodes" =~ ^[0-9]+$ ]] || { echo "Could not parse episode count from $log" >&2; exit 1; }
        expected_episodes=$((N_EPISODES_PER_TASK * 10))
        [[ "$episodes" -eq "$expected_episodes" ]] || {
            echo "Incomplete evaluation in $log: expected $expected_episodes episodes, got $episodes" >&2
            exit 1
        }
        echo "$(date +%Y-%m-%d),$RUN_ID,$EXP_NAME/$CHECKPOINT_STEP,$controller,$suite,$episodes,$sr" >> "$SUMMARY"
    done
done

echo "A/B evaluation complete: $SUMMARY"
cat "$SUMMARY"
