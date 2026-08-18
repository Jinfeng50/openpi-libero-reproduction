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

The current implementation is deliberately a Stage-A sidecar. It does not yet
alter action execution or claim a world-model control gain; that requires a
new, paired four-suite ablation after offline predictor and calibration checks.

On 2026-08-19, a real simulator smoke collected 10 Spatial episodes and 254
replan-boundary transitions, including both successful and failed episodes.
The episode shards, latent extraction, and one-epoch CPU critic fit all passed;
the compact tracked summary is
`experiments/world_model_stage_a_smoke.csv`. This is an implementation smoke,
not a learned-model result or a control benchmark.

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
