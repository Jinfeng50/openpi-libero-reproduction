# π0.5 LIBERO Official Checkpoint Baseline

## Protocol

Source: openpi `examples/libero/main.py` and `examples/libero/README.md`.

- Checkpoint: `gs://openpi-assets/checkpoints/pi05_libero`, mirrored locally at `/cfsdata/chenjinfeng/models/openpi/pi05_libero`
- Evaluation script: `examples/libero/main.py`
- Trials: 50 episodes per task
- Tasks: 10 tasks per suite
- Episodes: 500 episodes per suite, 2000 episodes total
- Seed: 7
- Replan steps: 5
- Image resize: 224

## Official Reference

| Model | Spatial | Object | Goal | Libero 10 | Average |
|---|:-:|:-:|:-:|:-:|:-:|
| π0.5 @ 30k official ckpt | 98.8 | 98.2 | 98.0 | 92.4 | 96.85 |

## My Previous Quick Check (Not Official Protocol)

This was run with 10 episodes per task, so it is only a sanity check.

| Suite | Episodes | Success Rate |
|---|---:|---:|
| Spatial | 100 | 99.0 |
| Object | 100 | 100.0 |
| Goal | 100 | 97.0 |
| Libero 10 | 100 | 94.0 |

## My Official-Protocol Reproduction

Completed run:

- Results: `/cfsdata/chenjinfeng/projects/openpi-libero-reproduction/experiments/baseline_official_50ep_20260607_1246`
- Videos: `/cfsdata/chenjinfeng/datasets/eval_videos/baseline_official_50ep_20260607_1246`

| Suite | Episodes | openpi Reference | Mine | Delta |
|---|---:|---:|---:|---:|
| Spatial | 500 | 98.8 | 98.2 | -0.6 |
| Object | 500 | 98.2 | 98.8 | +0.6 |
| Goal | 500 | 98.0 | 96.8 | -1.2 |
| Libero 10 | 500 | 92.4 | 92.6 | +0.2 |
| **Average** | 2000 | **96.85** | **96.60** | **-0.25** |

## Notes

- Do not pass `--args.num-trials-per-task 10` for official comparison.
- Evaluating official ckpt should use the checkpoint-bundled normalization stats.
- The final log line must contain `Total episodes: 500` for each suite.
- EGL cleanup warnings at process exit are harmless if total success rate and episode count were printed.
