# DGTE: Disagreement-Gated Temporal Ensemble

## Problem

The openpi LIBERO client requests a 10-step action chunk and executes five
actions before requesting another chunk. The second chunk predicts actions for
times that the first chunk already predicted, but the upstream client discards
those overlapping predictions. This can make the executed trajectory sensitive
to a single noisy inference at a replan boundary.

## Method

DGTE gives each candidate action an absolute simulator timestep. For a shared
timestep, older chunks receive an exponential freshness weight. If candidates
disagree, the newest chunk receives an additional bounded gate so the policy
can react quickly to a changed observation. Pose/action dimensions are blended;
the gripper dimension is copied from the newest candidate and is never averaged.

The controller is independent of JAX, the policy server, and robosuite. It is
therefore testable on CPU and can be reused with another action-chunk policy.
The implementation validates shapes and finite values, resets per episode, and
prunes chunks that are completely in the past.

## Reproducible ablation

```bash
EXP_NAME=pi05_libero_2gpu_20260610_1529 \
GPU_ID=<free-gpu> \
N_EPISODES_PER_TASK=50 \
./scripts/run_temporal_ablation.sh
```

The runner holds the checkpoint and server fixed, evaluates `baseline` and
`dgte` over `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`,
and stores logs/videos under `experiments/<run_id>/`. A quick 10-episode smoke
test is useful for catching simulator or server issues, but only the 50-episode
paired run belongs in the headline table.

For an eight-GPU node, the formal run used the equivalent parallel launcher:

```bash
EXP_NAME=pi05_libero_2gpu_20260610_1529 \
GPU_IDS="0 1 2 3 4 5 6 7" \
N_EPISODES_PER_TASK=50 \
./scripts/run_temporal_ablation_parallel.sh
```

## Formal result

The full paired benchmark completed on 2026-08-18. Each of the eight
controller/suite combinations contains 500 episodes. The observed mean success
rate changed from 95.75% to 96.35% (+0.60 percentage points): Spatial changed
from 96.8% to 97.8%, Object remained 97.6%, Goal changed from 96.8% to 96.6%,
and LIBERO-10 changed from 91.8% to 93.4%.

The paired episode outcomes contain 68 baseline-fail/DGTE-success cases and 56
baseline-success/DGTE-fail cases. A two-sided exact McNemar test gives
`p=0.323248`; the observed gain is therefore not statistically significant and
does not support a claim that DGTE reliably improves this benchmark. The
machine-readable success rates and paired counts are archived under
`experiments/dgte_ablation_parallel_20260818_2055_*`.
