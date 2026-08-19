# World Model + VLA Direction

This note turns recent world-model/VLA ideas into a tractable extension of the
current pi0.5 + LIBERO reproduction. It is a research plan, not a reported
result. The DGTE benchmark and the existing fine-tuned checkpoint remain the
controlled baselines.

## What transfers from the recent papers

| Idea | Useful lesson here | Scope risk |
|---|---|---|
| DeltaVLA | Anchor prediction to the current scene and model change rather than an absolute future image | SigLIP/DINOv2 pseudo-labels and multi-stage training add substantial dependencies |
| VLA-JEPA | Predict a future latent from the current observation and action without feeding the future frame to the encoder | A full Qwen/JEPA pretraining run is outside this reproduction budget |
| World-VLA-Loop | A reward/success head makes a world model useful for closed-loop policy selection | Cosmos-scale video generation is unnecessary for a first controlled experiment |
| VLA-MBPO | Short branched rollouts are more reliable than long open-loop imagination | Long-horizon model error must be measured and capped |

## Recommended contribution: latent change critic

Keep pi0.5 frozen and add a small action-conditioned critic beside the policy:

1. Encode the current camera observation into a compact latent `z_t` with a
   frozen image encoder. The current frame is the only input to the encoder.
2. Predict the future latent and a progress/success score from
   `(z_t, language, action_chunk)`. The target latent comes from a later frame,
   but is used only as a training target, so there is no future-information
   leakage at inference time.
3. At each replan boundary, score the overlapping action chunks with predicted
   progress minus calibrated uncertainty. Use the score to gate DGTE or fall
   back to the original newest chunk when the critic is uncertain.
4. Keep the imagined horizon short (one action chunk or two chunks) and
   re-anchor after every real observation. This is the practical MBPO/closed-
   loop part of the design.

This makes the novelty stronger than temporal smoothing: DGTE resolves action
disagreement, while the critic asks which candidate is more consistent with
the expected scene change and task progress.

## Minimal experiment ladder

### Stage A: data and predictor

Record synchronized `(image, wrist image, state, action chunk, language,
future image, success)` transitions from the existing LIBERO evaluator. Split
by task and initial state, not by adjacent frames. Train only the latent
predictor and reward head; report future-latent error, success calibration, and
CPU/GPU inference latency before touching policy control.

The implemented recorder writes one compressed NPZ shard per episode with a
schema version, current/future two-view images, states, the 10-step policy
chunk, five-step selected actions, actual executed-step count, terminal-within-
horizon, and episode success. Start collection with:

```bash
cd "$OPENPI_DIR"
PYTHONPATH="$PERSONAL_DIR/src:$LIBERO_DIR:$OPENPI_DIR/third_party/libero" \
  uv run python "$PERSONAL_DIR/scripts/eval_libero_temporal.py" \
  --task-suite-name libero_spatial --num-trials-per-task 50 \
  --controller baseline --record-transitions "$PERSONAL_DIR/experiments/world_model_spatial"
```

Frozen ResNet18 two-view latents can then be materialized and the critic trained
without feeding future images into the predictor:

```bash
cd "$OPENPI_DIR"
PYTHONPATH="$PERSONAL_DIR/src" uv run python "$PERSONAL_DIR/scripts/precompute_world_latents.py" \
  --data-dir "$PERSONAL_DIR/experiments/world_model_spatial" \
  --output-dir "$PERSONAL_DIR/experiments/world_model_spatial_latents"
PYTHONPATH="$PERSONAL_DIR/src" uv run python "$PERSONAL_DIR/scripts/train_world_model.py" \
  --data-dir "$PERSONAL_DIR/experiments/world_model_spatial_latents" \
  --output-dir "$PERSONAL_DIR/experiments/world_model_critic"
```

Evaluate the saved critic without simulator interaction. The evaluator reuses
the deterministic episode split, reports latent MSE, terminal/success AUROC,
Brier score, ten-bin expected calibration error, and measured model latency:

```bash
PYTHONPATH="$PERSONAL_DIR/src" uv run python "$PERSONAL_DIR/scripts/evaluate_world_model.py" \
  --data-dir "$PERSONAL_DIR/experiments/world_model_spatial_latents" \
  --checkpoint "$PERSONAL_DIR/experiments/world_model_critic/critic.pt" \
  --device cuda:4 \
  --output-json "$PERSONAL_DIR/experiments/world_model_critic/evaluation.json"
```

The trained Stage-A critic is deliberately kept separate from the reported
baseline and DGTE results. The optional inference-only controller below can
alter action selection, but it does not claim a world-model control gain until
a new, paired four-suite ablation is completed.

An inference-only `world_model` controller is now available for that next
ablation. At each real replan boundary it scores the currently overlapping full
policy chunks using the current two-view observation, state, language, and the
critic; it executes the selected chunk slice and never supplies future images
to the selector. Historical chunks are aligned on the absolute simulator
timestep before slicing.

Run a small Spatial smoke against a local checkpoint with:

```bash
PYTHONPATH="$PERSONAL_DIR/src:$LIBERO_DIR:$OPENPI_DIR/third_party/libero" \
uv run python "$PERSONAL_DIR/scripts/eval_libero_temporal.py" \
  --task-suite-name libero_spatial --num-trials-per-task 1 \
  --controller world_model \
  --world-model-checkpoint "$PERSONAL_DIR/experiments/world_model_pilot_20260819_0740_imagenet_critic/critic.pt" \
  --world-model-device cuda:1 --world-model-encoder-weights default
```

The 2026-08-19 ten-episode Spatial smoke completed at 10/10 and produced ten
transition shards under the ignored `experiments/world_model_control_smoke_*`
directory. The compact tracked record is
`experiments/world_model_control_smoke_summary.csv`. This validates the
inference path only; it is not a paired control benchmark and does not
establish a gain over baseline or DGTE.

That simulator smoke preceded the final correction that aligns a historical
chunk's critic input to the current absolute timestep. The corrected logic has
passed unit tests and replay over real recorded transitions, but has not yet
been rerun in the simulator because all eight GPUs are occupied by an existing
training job. The smoke therefore remains interface evidence, not validation
of the final scoring rule.

The reproducible eight-GPU paired runner is
`scripts/run_world_model_ablation_parallel.sh`. It runs baseline and
world-model controllers with the same seed and episode protocol, records
episode shards for both, and refuses to start if the local ResNet18 weight file
is missing. Use a small pilot first:

```bash
EXP_NAME=pi05_libero_2gpu_20260610_1529 \
WORLD_MODEL_CHECKPOINT="$PERSONAL_DIR/experiments/world_model_pilot_20260819_0740_imagenet_critic/critic.pt" \
N_EPISODES_PER_TASK=10 \
./scripts/run_world_model_ablation_parallel.sh
```

On 2026-08-19, a real simulator smoke collected 10 Spatial episodes and 254
replan-boundary transitions, including both successful and failed episodes.
The episode shards, latent extraction, and one-epoch CPU critic fit all passed;
the compact tracked summary is
`experiments/world_model_stage_a_smoke.csv`. This is an implementation smoke,
not a learned-model result or a control benchmark.

### Stage-A pilot result (offline only)

The subsequent baseline pilot collected 100 episodes per suite (400 episodes,
13,140 transitions) from the fine-tuned checkpoint at seed 7. The simulator
success rates were Spatial 97%, Object 96%, Goal 95%, and LIBERO-10 91%. These
numbers describe the data-collection rollouts, not a new policy benchmark.

Two frozen ResNet18 representations were compared using the same deterministic
episode split (264 train, 89 validation, 47 test episodes). The first attempt
to use ImageNet weights timed out while downloading from
`download.pytorch.org`; no substitute was invented. After the weight file was
provided locally, the ImageNet run completed. Test-set metrics are:

| Frozen representation | Latent MSE | Terminal AUROC | Success AUROC | Success Brier | Critic-only latency |
|---|---:|---:|---:|---:|---:|
| Randomly initialized ResNet18 | 0.00401 | 0.8535 | 0.7934 | 0.1234 | 20.30 us/sample |
| ImageNet ResNet18 | 0.10854 | 0.9551 | 0.8732 | 0.1146 | 6.37 us/sample |

Latent MSE is meaningful only within a fixed representation because feature
scales differ across encoders; it must not be used to rank the two rows. The
latency column measures only the critic on precomputed latents, excludes the
ResNet encoder, and was collected on a shared GPU, so it is descriptive rather
than a controlled systems comparison.

The complete machine-readable summary is
`experiments/world_model_pilot_20260819_0740_evaluation_summary.csv`; the
full JSON reports and checkpoints remain outside Git under the corresponding
`experiments/world_model_pilot_20260819_0740_*` directories. These are offline
predictor/calibration measurements only. The critic has not changed action
execution, and no world-model control gain is claimed.

### Stage B: inference-only control

Compare the same checkpoint and seed under:

- vanilla openpi client;
- DGTE;
- DGTE + latent change critic (the proposed method);
- critic gating disabled, to isolate the value of the world-model signal.

Use 50 episodes per task on all four LIBERO suites. Add visual-corruption and
initial-state perturbation slices because a world model should help most when
the current observation differs from the training distribution.

### Stage C: optional learned adapter

Only if Stage B is positive, fine-tune a small action-head/adapter with the
critic loss as an auxiliary objective. Keep the original full fine-tune and
the inference-only result as separate rows; do not mix normalization files or
claim gains without the same paired protocol.

## Falsifiable claims and gates

The project should claim an improvement only if the critic improves paired
success rate (or robustness) with a confidence interval, while adding a
bounded per-step latency and no increase in failure on the clean LIBERO
baseline. If the predictor has low offline error but no closed-loop gain, that
is still a useful negative result: it distinguishes representation quality
from control utility.

The first implementation should therefore be a small sidecar and an explicit
ablation, not a new foundation-model pretraining claim. Full video generation,
JEPA pretraining, and model-based RL can remain follow-up work after this
controlled result is reproducible.
