# Datasets

## LIBERO
- Path: `/cfsdata/chenjinfeng/datasets/libero` (linked to `physical-intelligence/libero`)
- Format: LeRobotDataset
- Stats: 1693 episodes, 273465 frames, action_dim=7
- Features: actions, state, image, wrist_image, task_index, episode_index, frame_index, timestamp, index

## Generated norm_stats
- Path: `/cfsdata/chenjinfeng/projects/openpi/assets/pi05_libero/physical-intelligence/libero/norm_stats.json`
- Contains: state/actions mean, std, q01, q99
- State stats dim: 8
- Action stats dim: 7

## norm_stats Strategy
- Purpose: normalize LIBERO state/action dimensions so training sees stable numeric scales.
- From base fine-tuning (`pi05_base` → LIBERO): use freshly computed stats from `compute_norm_stats.py`.
- Evaluating official `pi05_libero` checkpoint: use checkpoint-bundled stats.
- Never mix these two cases; wrong stats distort action denormalization and can collapse SR.
