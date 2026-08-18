#!/usr/bin/env bash
# Collect episode-safe transition shards for the Stage-A world-model sidecar.
# One baseline evaluator/server is assigned to each LIBERO suite.
set -euo pipefail

OPENPI_DIR=${OPENPI_DIR:-/cfsdata/chenjinfeng/projects/openpi}
PERSONAL_DIR=${PERSONAL_DIR:-/cfsdata/chenjinfeng/projects/openpi-libero-reproduction}
LIBERO_DIR=${LIBERO_DIR:-/cfsdata/chenjinfeng/projects/LIBERO}
CONFIG=${CONFIG:-pi05_libero}
EXP_NAME=${EXP_NAME:?Set EXP_NAME to the checkpoint experiment name}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-29999}
N_EPISODES_PER_TASK=${N_EPISODES_PER_TASK:-10}
SERVER_PORT_BASE=${SERVER_PORT_BASE:-8500}
GPU_IDS=${GPU_IDS:-"0 1 2 3"}
RUN_ID=${RUN_ID:-world_model_data_$(date +%Y%m%d_%H%M)}
EXP_DIR="$PERSONAL_DIR/experiments/$RUN_ID"
CKPT="$OPENPI_DIR/checkpoints/$CONFIG/$EXP_NAME/$CHECKPOINT_STEP"
TRANSITIONS_DIR="$EXP_DIR/transitions"

if [[ ! -d "$CKPT" ]]; then
    echo "Checkpoint does not exist: $CKPT" >&2
    exit 1
fi
read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -lt 4 ]]; then
    echo "GPU_IDS must provide at least four GPU IDs" >&2
    exit 1
fi

export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/cfsdata/chenjinfeng/openpi_cache}
export HF_HOME=${HF_HOME:-/cfsdata/chenjinfeng/hf_cache}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/cfsdata/chenjinfeng/datasets}
export TMPDIR=${TMPDIR:-/cfsdata/chenjinfeng/tmp}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH="$LIBERO_DIR:$OPENPI_DIR/third_party/libero:$PERSONAL_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$EXP_DIR/logs" "$EXP_DIR/videos" "$TRANSITIONS_DIR"
SUMMARY="$EXP_DIR/collection_summary.csv"
echo "date,run_id,checkpoint,controller,suite,episodes,success_rate,transitions" > "$SUMMARY"

declare -a SERVER_PIDS=()
declare -a JOB_PIDS=()
declare -a JOB_NAMES=()
cleanup() {
    local pid
    for pid in "${SERVER_PIDS[@]:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${SERVER_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

server_ready() {
    local port=$1
    local log=$2
    grep -q "server listening on 0.0.0.0:$port" "$log" 2>/dev/null
}

start_server() {
    local gpu=$1
    local port=$2
    local log=$3
    (
        cd "$OPENPI_DIR"
        CUDA_VISIBLE_DEVICES="$gpu" uv run scripts/serve_policy.py \
            --port "$port" \
            policy:checkpoint \
            --policy.config="$CONFIG" \
            --policy.dir="$CKPT"
    ) > "$log" 2>&1 &
    SERVER_PIDS+=("$!")
}

run_job() {
    local suite=$1
    local port=$2
    local log="$EXP_DIR/logs/baseline_$suite.log"
    local video_dir="$EXP_DIR/videos/baseline/$suite"
    mkdir -p "$video_dir"
    (
        cd "$OPENPI_DIR"
        uv run python "$PERSONAL_DIR/scripts/eval_libero_temporal.py" \
            --host localhost \
            --port "$port" \
            --task-suite-name "$suite" \
            --num-trials-per-task "$N_EPISODES_PER_TASK" \
            --video-out-path "$video_dir" \
            --controller baseline \
            --record-transitions "$TRANSITIONS_DIR"
    ) > "$log" 2>&1 &
    JOB_PIDS+=("$!")
    JOB_NAMES+=("$suite")
}

declare -a SUITES=(libero_spatial libero_object libero_goal libero_10)
for index in "${!SUITES[@]}"; do
    suite="${SUITES[$index]}"
    start_server "${GPUS[$index]}" "$((SERVER_PORT_BASE + index))" \
        "$EXP_DIR/logs/server_$suite.log"
done

for index in "${!SERVER_PIDS[@]}"; do
    suite="${SUITES[$index]}"
    port=$((SERVER_PORT_BASE + index))
    log="$EXP_DIR/logs/server_$suite.log"
    ready=0
    for _ in $(seq 1 240); do
        if server_ready "$port" "$log"; then
            ready=1
            break
        fi
        if ! kill -0 "${SERVER_PIDS[$index]}" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    if [[ "$ready" -ne 1 ]]; then
        echo "Policy server on port $port failed; see $log" >&2
        exit 1
    fi
done

for index in "${!SUITES[@]}"; do
    run_job "${SUITES[$index]}" "$((SERVER_PORT_BASE + index))"
done

failed=0
for index in "${!JOB_PIDS[@]}"; do
    if ! wait "${JOB_PIDS[$index]}"; then
        echo "Evaluation failed: ${JOB_NAMES[$index]}" >&2
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    exit 1
fi

for suite in "${SUITES[@]}"; do
    log="$EXP_DIR/logs/baseline_$suite.log"
    sr=$(awk '/Total success rate:/ { value=$NF } END { print value }' "$log")
    episodes=$(awk '/Total episodes:/ { value=$NF } END { print value }' "$log")
    [[ "$sr" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Could not parse SR from $log" >&2; exit 1; }
    [[ "$episodes" =~ ^[0-9]+$ ]] || { echo "Could not parse episode count from $log" >&2; exit 1; }
    expected=$((N_EPISODES_PER_TASK * 10))
    [[ "$episodes" -eq "$expected" ]] || {
        echo "Incomplete $suite: expected $expected episodes, got $episodes" >&2
        exit 1
    }
    transitions=$(python3 - "$TRANSITIONS_DIR" "$suite" <<'PY'
import pathlib
import sys
import numpy as np

root = pathlib.Path(sys.argv[1])
suite = sys.argv[2]
total = 0
for path in root.glob(f"{suite}_*.npz"):
    with np.load(path, allow_pickle=False) as data:
        total += len(data["image"])
print(total)
PY
)
    echo "$(date +%Y-%m-%d),$RUN_ID,$EXP_NAME/$CHECKPOINT_STEP,baseline,$suite,$episodes,$sr,$transitions" >> "$SUMMARY"
done

echo "Transition collection complete: $EXP_DIR"
cat "$SUMMARY"
