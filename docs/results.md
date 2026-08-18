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

`baseline` and `dgte` rows will be appended only from the paired runner. No
placeholder or unverified success rate is intentionally kept here.
