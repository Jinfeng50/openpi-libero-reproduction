# World Model + VLA Direction

This note turns recent world-model/VLA ideas into a tractable extension of the
current pi0.5 + LIBERO reproduction. The sidecar implementation and its first
two paired pilots are now reported here. The DGTE benchmark and the existing
fine-tuned checkpoint remain the controlled baselines.

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
chunk's critic input to the current absolute timestep. The correction also
selects the five actions actually executed before the recorded future frame,
instead of training on the full ten-action policy chunk. The aligned critic
passed the unit suite, CUDA loading/smoke checks, and a fresh paired simulator
pilot below.

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

The same runner can compare DGTE against the confidence-gated hybrid without
changing the script:

```bash
CONTROLLERS="dgte hybrid" \
WORLD_MODEL_CHECKPOINT="$PERSONAL_DIR/experiments/world_model_pilot_20260819_0740_imagenet_aligned_critic/critic.pt" \
WORLD_MODEL_GATE_MARGIN=0.001 \
WORLD_MODEL_GATE_UNCERTAINTY=0.40 \
N_EPISODES_PER_TASK=10 \
./scripts/run_world_model_ablation_parallel.sh
```

### Stage-B pilot result

The paired pilot completed on 2026-08-19 with 100 episodes per task on all
four suites for each controller. The same checkpoint, seed, task order, and
episode indices were used, and the paired keys matched exactly:

| Suite | Baseline | World-model | Delta |
|---|---:|---:|---:|
| Spatial | 99% | 96% | -3pp |
| Object | 96% | 97% | +1pp |
| Goal | 97% | 93% | -4pp |
| LIBERO-10 | 91% | 91% | 0pp |
| Aggregate (400) | 95.75% | 94.25% | -1.50pp |

The exact paired McNemar test over all 400 episodes was `p=0.4050` (21
baseline-only wins versus 15 world-model-only wins). This pilot therefore does
not support a world-model control gain. The result is useful as a negative
control-selection finding: strong offline AUROC did not transfer to this
first closed-loop selector. The full compact record is
`experiments/world_model_ablation_pilot_20260819_0920_summary.csv`; raw videos,
transition shards, and logs remain in the ignored experiment directory.

### Stage-B aligned pilot result

The critic was retrained with `selected_actions`, the five actions executed
before each future observation. Its held-out test success AUROC was `0.8957`
(versus `0.8732` for the original misaligned ImageNet critic); terminal AUROC
was `0.9403`, success Brier was `0.1040`, and latent MSE was `0.108636`.
These are offline predictor metrics and do not measure policy success.

A fresh paired four-suite pilot used the same checkpoint, seed, task order, and
100 episodes per task for each controller:

| Suite | Baseline | World-model | Delta |
|---|---:|---:|---:|
| Spatial | 98% | 97% | -1pp |
| Object | 99% | 99% | 0pp |
| Goal | 97% | 92% | -5pp |
| LIBERO-10 | 92% | 93% | +1pp |
| Aggregate (400) | 96.50% | 95.25% | -1.25pp |

The exact paired McNemar test was `p=0.4421` (16 baseline-only wins versus 11
world-model-only wins). The aligned critic therefore still does not support a
closed-loop control gain. The result is retained as a useful negative finding:
better action/target alignment improved offline success discrimination but did
not transfer to this inference-only chunk selector. The compact record is
`experiments/world_model_aligned_pilot_20260819_1100_summary.csv`; raw videos,
transition shards, and logs remain outside Git.

### DGTE + world-model confidence gate

To test whether the negative selector result came from overconfident
interventions, the client now supports `--controller hybrid`. It computes the
DGTE action and the world-model candidate scores at each replan boundary. The
world-model choice is used only when the score margin is at least `0.001` and
the best-candidate latent uncertainty is at most `0.40`; otherwise it falls
back to DGTE. The runner exposes these thresholds through
`WORLD_MODEL_GATE_MARGIN` and `WORLD_MODEL_GATE_UNCERTAINTY` and logs the
acceptance count.

The paired pilot used 100 episodes per task on all four suites. Gate acceptance
was 18.3% on Spatial, 29.2% on Object, 30.1% on Goal, and 39.2% on LIBERO-10.
DGTE reached 97.00% overall while the hybrid reached 95.00% (`-2.00pp`, exact
paired McNemar `p=0.1338`; 15 DGTE-only wins versus 7 hybrid-only wins). The
confidence gate therefore did not rescue the control result and is retained as
a negative ablation. The compact record is
`experiments/world_model_hybrid_smoke_20260819_1300_summary.csv`; raw videos,
transition shards, and logs remain outside Git.

For the next run, transition shards also store optional per-replan gate
metadata (`gate_accepted`, `gate_margin`, `gate_uncertainty`, and
`gate_candidate_count`). Existing schema-1 shards remain readable because
these fields are additive. This enables a post-hoc threshold sweep grouped by
task and episode instead of relying only on aggregate gate counts.
Run `scripts/analyze_gate_metadata.py` on one or more completed hybrid
transition roots to produce the merged percentile and threshold-sweep JSON.

A first metadata-enabled Spatial/Goal run was interrupted after 294 of 400
planned shards when two `uv run` wrappers lost their evaluator child. Its
partial data is diagnostic-only and is not used for a control claim; the
tracked status is
`experiments/world_model_gate_metadata_pilot_20260819_2209_status.csv`.

After hardening the runner with per-job timeouts and evaluator cleanup, a
16-episode-per-controller Spatial/Goal smoke completed cleanly. The hybrid
accepted the world-model choice on 4.5% of replan boundaries and achieved
39/40 successes versus DGTE's 38/40; this is an interface smoke with only 40
paired episodes (`p=1.0`), not evidence of a control gain. The compact record
is `experiments/world_model_gate_metadata_smoke_20260820_1535_summary.csv`.

### Full metadata-enabled gate result

The hardened runner then completed two four-way batches without touching the
user-owned process on GPU2: Spatial/Object in
`world_model_gate_full_20260820_spatial_object` and Goal/LIBERO-10 in
`world_model_gate_full_20260820_goal_10`. Each batch used DGTE and the hybrid
controller with the same seed, task order, checkpoint, and 100 episodes per
suite. The merged result is:

| Suite | DGTE | Hybrid | Delta |
|---|---:|---:|---:|
| Spatial | 98/100 | 100/100 | +2pp |
| Object | 99/100 | 99/100 | 0pp |
| Goal | 99/100 | 100/100 | +1pp |
| LIBERO-10 | 93/100 | 94/100 | +1pp |
| Aggregate (400) | 389/400 (97.25%) | 393/400 (98.25%) | +1pp |

The paired exact McNemar test is `p=0.3876953125` (4 DGTE-only wins versus 8
hybrid-only wins), so this result does not establish a statistically reliable
control gain. The configured gate accepted 1,258 of 12,788 candidate
transitions (9.84%) across the four suites. Acceptance by suite was 4.10%
(Spatial), 7.54% (Object), 6.20% (Goal), and 14.89% (LIBERO-10). These are
transition-level observational rates; accepted-transition success is not a
counterfactual estimate of the DGTE action. The merged machine-readable files
are `experiments/world_model_gate_full_20260820_four_suite_sr_summary.csv`,
`experiments/world_model_gate_full_20260820_four_suite_paired_counts.csv`, and
`experiments/world_model_gate_full_20260820_four_suite_gate_metadata.json`.

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
| ImageNet ResNet18, original 10-step input | 0.10854 | 0.9551 | 0.8732 | 0.1146 | 6.37 us/sample |
| ImageNet ResNet18, aligned 5-step input | 0.108636 | 0.9403 | 0.8957 | 0.1040 | 2.62 us/sample |

Latent MSE is meaningful only within a fixed representation because feature
scales differ across encoders; it must not be used to rank the two rows. The
latency column measures only the critic on precomputed latents, excludes the
ResNet encoder, and was collected on a shared GPU, so it is descriptive rather
than a controlled systems comparison.

The complete machine-readable summaries are
`experiments/world_model_pilot_20260819_0740_evaluation_summary.csv` and
`experiments/world_model_pilot_20260819_0740_aligned_evaluation_summary.csv`;
the full JSON reports and checkpoints remain outside Git under the corresponding
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
