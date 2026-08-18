# Results Ledger

## Official checkpoint reproduction

Protocol: openpi `examples/libero/main.py`, seed 7, replan steps 5, 50
episodes per task, 10 tasks per suite. Full logs are in
`experiments/baseline_official_50ep_20260607_1246/`.

| Suite | Episodes | Success rate |
|---|---:|---:|
| `libero_spatial` | 500 | 0.982 |
| `libero_object` | 500 | 0.988 |
| `libero_goal` | 500 | 0.968 |
| `libero_10` | 500 | 0.926 |
| **Average** | **2,000** | **0.9660** |

## Full fine-tuning

Run: `pi05_libero_2gpu_20260610_1529`
Checkpoint: `/cfsdata/chenjinfeng/projects/openpi/checkpoints/pi05_libero/pi05_libero_2gpu_20260610_1529/29999`
Evaluation: `experiments/post_train_eval_20260612_1120/`

| Suite | Episodes | Success rate |
|---|---:|---:|
| `libero_spatial` | 500 | 0.984 |
| `libero_object` | 500 | 0.984 |
| `libero_goal` | 500 | 0.968 |
| `libero_10` | 500 | 0.918 |
| **Average** | **2,000** | **0.9635** |

The run took approximately 28.4 hours including checkpoint finalization. The
single `EGL_NOT_INITIALIZED` cleanup message occurred after the final success
rate and episode count had been printed; the logs contain all 500 episodes for
each suite.

## DGTE

The formal paired run used the fine-tuned `29999` checkpoint, seed 7, replan
steps 5, 50 episodes per task, and 10 tasks per suite. Baseline and DGTE were
evaluated independently with the same protocol:

| Controller | Spatial | Object | Goal | LIBERO-10 | Average |
|---|---:|---:|---:|---:|---:|
| baseline | 0.968 | 0.976 | 0.968 | 0.918 | **0.9575** |
| DGTE | 0.978 | 0.976 | 0.966 | 0.934 | **0.9635** |
| DGTE - baseline | +0.010 | 0.000 | -0.002 | +0.016 | **+0.0060** |

Every controller/suite row contains 500 episodes (4,000 episodes total). The
tracked source is
`experiments/dgte_ablation_parallel_20260818_2055_sr_summary.csv`; full logs and
videos remain under the corresponding ignored experiment directory. The
observed 0.60-point mean gain is not evidence of uniform improvement: Goal
decreased slightly and Object was unchanged.

Pairing outcomes by task, initial-state index, and episode order gives 68 cases
where baseline failed and DGTE succeeded versus 56 cases in the opposite
direction. The two-sided exact McNemar test gives `p=0.323248`, so the aggregate
difference is not statistically significant at 0.05. Suite-level paired counts
are archived in
`experiments/dgte_ablation_parallel_20260818_2055_paired_counts.csv`.

The baseline values in this paired run should be used for the DGTE delta. They
are kept separate from the earlier fine-tuning evaluation because that run
used a different evaluator process and execution schedule.
