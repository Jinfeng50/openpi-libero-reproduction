#!/usr/bin/env bash
# Paired baseline vs inference-only world-model controller ablation.
set -euo pipefail

OPENPI_DIR=${OPENPI_DIR:-/cfsdata/chenjinfeng/projects/openpi}
PERSONAL_DIR=${PERSONAL_DIR:-/cfsdata/chenjinfeng/projects/openpi-libero-reproduction}
LIBERO_DIR=${LIBERO_DIR:-/cfsdata/chenjinfeng/projects/LIBERO}
CONFIG=${CONFIG:-pi05_libero}
EXP_NAME=${EXP_NAME:?Set EXP_NAME to the checkpoint experiment name}
CHECKPOINT_STEP=${CHECKPOINT_STEP:-29999}
WORLD_MODEL_CHECKPOINT=${WORLD_MODEL_CHECKPOINT:?Set WORLD_MODEL_CHECKPOINT to critic.pt}
WORLD_MODEL_ENCODER_WEIGHTS=${WORLD_MODEL_ENCODER_WEIGHTS:-default}
WORLD_MODEL_GATE_MARGIN=${WORLD_MODEL_GATE_MARGIN:-0.001}
WORLD_MODEL_GATE_UNCERTAINTY=${WORLD_MODEL_GATE_UNCERTAINTY:-0.40}
N_EPISODES_PER_TASK=${N_EPISODES_PER_TASK:-10}
SERVER_PORT_BASE=${SERVER_PORT_BASE:-8700}
GPU_IDS=${GPU_IDS:-"0 1 2 3 4 5 6 7"}
RUN_ID=${RUN_ID:-world_model_ablation_parallel_$(date +%Y%m%d_%H%M)}
EXP_DIR="$PERSONAL_DIR/experiments/$RUN_ID"
CKPT="$OPENPI_DIR/checkpoints/$CONFIG/$EXP_NAME/$CHECKPOINT_STEP"

if [[ ! -d "$CKPT" ]]; then
    echo "Checkpoint does not exist: $CKPT" >&2
    exit 1
fi
if [[ ! -f "$WORLD_MODEL_CHECKPOINT" ]]; then
    echo "World-model checkpoint does not exist: $WORLD_MODEL_CHECKPOINT" >&2
    exit 1
fi
if [[ "$WORLD_MODEL_ENCODER_WEIGHTS" == "default" ]]; then
    TORCH_CACHE_ROOT=${TORCH_HOME:-/cfsdata/chenjinfeng/.cache/torch}
    RESNET_WEIGHTS="$TORCH_CACHE_ROOT/hub/checkpoints/resnet18-f37072fd.pth"
    if [[ ! -f "$RESNET_WEIGHTS" ]]; then
        echo "Missing ResNet18 weights: $RESNET_WEIGHTS" >&2
        echo "Download manually from https://download.pytorch.org/models/resnet18-f37072fd.pth" >&2
        exit 1
    fi
fi

read -r -a GPUS <<< "$GPU_IDS"
if [[ "$N_EPISODES_PER_TASK" -le 0 ]]; then
    echo "N_EPISODES_PER_TASK must be positive" >&2
    exit 1
fi

export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/cfsdata/chenjinfeng/openpi_cache}
export HF_HOME=${HF_HOME:-/cfsdata/chenjinfeng/hf_cache}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/cfsdata/chenjinfeng/datasets}
export TMPDIR=${TMPDIR:-/cfsdata/chenjinfeng/tmp}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH="$LIBERO_DIR:$OPENPI_DIR/third_party/libero:$PERSONAL_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$EXP_DIR/logs" "$EXP_DIR/videos" "$EXP_DIR/transitions"
SUMMARY="$EXP_DIR/sr_summary.csv"
echo "date,run_id,checkpoint,controller,suite,episodes,success_rate" > "$SUMMARY"

read -r -a CONTROLLERS <<< "${CONTROLLERS:-baseline world_model}"
for controller in "${CONTROLLERS[@]}"; do
    case "$controller" in
        baseline|dgte|world_model|hybrid) ;;
        *) echo "Unsupported controller: $controller" >&2; exit 1 ;;
    esac
done
if [[ "${#CONTROLLERS[@]}" -eq 0 ]]; then
    echo "CONTROLLERS must contain at least one controller" >&2
    exit 1
fi

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
            --port "$port" policy:checkpoint \
            --policy.config="$CONFIG" --policy.dir="$CKPT"
    ) > "$log" 2>&1 &
    SERVER_PIDS+=("$!")
}

run_job() {
    local controller=$1
    local suite=$2
    local port=$3
    local gpu=$4
    local log="$EXP_DIR/logs/${controller}_${suite}.log"
    local video_dir="$EXP_DIR/videos/${controller}/${suite}"
    local transition_dir="$EXP_DIR/transitions/${controller}"
    mkdir -p "$video_dir" "$transition_dir"
    (
        cd "$OPENPI_DIR"
        local -a command=(
            uv run python "$PERSONAL_DIR/scripts/eval_libero_temporal.py"
            --host localhost --port "$port"
            --task-suite-name "$suite"
            --num-trials-per-task "$N_EPISODES_PER_TASK"
            --video-out-path "$video_dir"
            --controller "$controller"
            --record-transitions "$transition_dir"
        )
        if [[ "$controller" == "world_model" || "$controller" == "hybrid" ]]; then
            command+=(
                --world-model-checkpoint "$WORLD_MODEL_CHECKPOINT"
                --world-model-device "cuda:$gpu"
                --world-model-encoder-weights "$WORLD_MODEL_ENCODER_WEIGHTS"
            )
            if [[ "$controller" == "hybrid" ]]; then
                command+=(
                    --world-model-gate-margin "$WORLD_MODEL_GATE_MARGIN"
                    --world-model-gate-uncertainty "$WORLD_MODEL_GATE_UNCERTAINTY"
                )
            fi
        fi
        "${command[@]}"
    ) > "$log" 2>&1 &
    JOB_PIDS+=("$!")
    JOB_NAMES+=("$controller/$suite")
}

declare -a SUITES=(libero_spatial libero_object libero_goal libero_10)
required_gpus=$((${#CONTROLLERS[@]} * ${#SUITES[@]}))
if [[ "${#GPUS[@]}" -lt "$required_gpus" ]]; then
    echo "GPU_IDS must provide at least $required_gpus GPU IDs for ${#CONTROLLERS[@]} controllers (got ${#GPUS[@]})" >&2
    exit 1
fi
job_index=0
for controller in "${CONTROLLERS[@]}"; do
    for suite in "${SUITES[@]}"; do
        start_server "${GPUS[$job_index]}" "$((SERVER_PORT_BASE + job_index))" \
            "$EXP_DIR/logs/server_${controller}_${suite}.log"
        job_index=$((job_index + 1))
    done
done

for index in "${!SERVER_PIDS[@]}"; do
    port=$((SERVER_PORT_BASE + index))
    controller="${CONTROLLERS[$((index / ${#SUITES[@]}))]}"
    suite="${SUITES[$((index % ${#SUITES[@]}))]}"
    log="$EXP_DIR/logs/server_${controller}_${suite}.log"
    ready=0
    for _ in $(seq 1 240); do
        if server_ready "$port" "$log"; then ready=1; break; fi
        if ! kill -0 "${SERVER_PIDS[$index]}" 2>/dev/null; then break; fi
        sleep 1
    done
    if [[ "$ready" -ne 1 ]]; then
        echo "Policy server on port $port failed; see $log" >&2
        exit 1
    fi
done

job_index=0
for controller in "${CONTROLLERS[@]}"; do
    for suite in "${SUITES[@]}"; do
        run_job "$controller" "$suite" "$((SERVER_PORT_BASE + job_index))" "${GPUS[$job_index]}"
        job_index=$((job_index + 1))
    done
done

failed=0
for index in "${!JOB_PIDS[@]}"; do
    if ! wait "${JOB_PIDS[$index]}"; then
        echo "Evaluation failed: ${JOB_NAMES[$index]}" >&2
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then exit 1; fi

for controller in "${CONTROLLERS[@]}"; do
    for suite in "${SUITES[@]}"; do
        log="$EXP_DIR/logs/${controller}_${suite}.log"
        sr=$(awk '/Total success rate:/ { value=$NF } END { print value }' "$log")
        episodes=$(awk '/Total episodes:/ { value=$NF } END { print value }' "$log")
        [[ "$sr" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "Could not parse SR from $log" >&2; exit 1; }
        expected=$((N_EPISODES_PER_TASK * 10))
        [[ "$episodes" == "$expected" ]] || { echo "Incomplete $log: got $episodes" >&2; exit 1; }
        echo "$(date +%Y-%m-%d),$RUN_ID,$EXP_NAME/$CHECKPOINT_STEP,$controller,$suite,$episodes,$sr" >> "$SUMMARY"
    done
done

echo "World-model paired ablation complete: $SUMMARY"
cat "$SUMMARY"
if [[ "${#CONTROLLERS[@]}" -eq 2 ]]; then
    uv run python "$PERSONAL_DIR/scripts/analyze_world_model_ablation.py" \
        --transition-root "$EXP_DIR/transitions" \
        --baseline-controller "${CONTROLLERS[0]}" \
        --world-model-controller "${CONTROLLERS[1]}" \
        --output-csv "$EXP_DIR/paired_counts.csv"
else
    echo "Skipping paired-count analysis: expected exactly two controllers, got ${#CONTROLLERS[@]}" >&2
fi
