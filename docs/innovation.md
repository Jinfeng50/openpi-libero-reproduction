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

## Current status

The implementation and CPU unit tests are complete. A real simulator smoke run
also completed successfully on 2026-08-18: one episode per task on
`libero_spatial` gave 10/10 for both baseline and DGTE. This is only an
integration check, not a headline result. The four-suite, 50-episode-per-task
paired benchmark remains pending, and no DGTE improvement is claimed before
that controlled run finishes.
