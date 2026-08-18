<div align="center">

# pi0.5 LIBERO Reproduction and DGTE Inference

Reproducible training and evaluation records for Physical Intelligence's
openpi pi0.5 policy on LIBERO, plus a training-free temporal action fusion
baseline for controlled ablations.

[Results](#results) | [Reproduction](#reproduction) | [DGTE](#innovation-dgte) | [World-model direction](docs/world_model_direction.md)

</div>

## Scope

This repository records two completed experiments and one executable research
extension:

1. The official `pi05_libero` checkpoint was evaluated with the openpi LIBERO
   client using 50 episodes per task and seed 7.
2. `pi05_base` was fully fine-tuned on LIBERO for 30,000 steps with FSDP over
   2x A800-80GB GPUs.
3. DGTE (Disagreement-Gated Temporal Ensemble) is implemented as a client-side
   inference method. It fuses overlapping action chunks without changing the
   checkpoint or requiring another training run. Its paired four-suite
   benchmark completed with a +0.60 percentage-point mean change.

The repository does not contain model weights. Checkpoint paths in the tables
are local archive paths from the original run; use the commands below to point
the scripts at your own copy.

## Results

All completed numbers use 10 tasks per suite, 50 episodes per task (500
episodes per suite), and seed 7. Logs and CSV files are under
[`experiments/`](experiments/).

| Method | Spatial | Object | Goal | LIBERO-10 | Average |
|---|---:|---:|---:|---:|---:|
| Official pi0.5 checkpoint, reproduced | 98.2% | 98.8% | 96.8% | 92.6% | **96.60%** |
| pi05_base -> LIBERO, 2x A800, 30k steps | 98.4% | 98.4% | 96.8% | 91.8% | **96.35%** |
| Fine-tuned checkpoint, paired DGTE baseline | 96.8% | 97.6% | 96.8% | 91.8% | **95.75%** |
| DGTE on the fine-tuned checkpoint | 97.8% | 97.6% | 96.6% | 93.4% | **96.35%** |

The official checkpoint row is independently documented in
[`docs/baseline.md`](docs/baseline.md). The fine-tuning row is archived in
[`experiments/results.csv`](experiments/results.csv) and
[`docs/results.md`](docs/results.md). No OOD or multimodal result is claimed in
this repository yet.

The formal paired DGTE run used 500 episodes per suite for each controller.
The observed four-suite mean was 0.60 percentage points above its paired
baseline, with gains on Spatial and LIBERO-10, no change on Object, and a
0.20-point decrease on Goal. The aggregate exact McNemar test was not
significant (`p=0.323`), so this is not evidence of a reliable overall gain.
The tracked summary is
[`experiments/dgte_ablation_parallel_20260818_2055_sr_summary.csv`](experiments/dgte_ablation_parallel_20260818_2055_sr_summary.csv).

Training evidence is kept in [`docs/figures/`](docs/figures/) and the linked
WandB run: <https://wandb.ai/3267189544-uestc/openpi/runs/hwdxnlvn>.

The checked-in comparison figure is generated with
`python scripts/plot_results.py`.

The next research extension is documented in
[`docs/world_model_direction.md`](docs/world_model_direction.md): a frozen
pi0.5 policy plus a short-horizon latent change critic and progress head. It is
explicitly a plan until its paired LIBERO ablation is complete.

## Reproduction

### Environment

The recorded machine used Ubuntu 22.04, Python 3.11.14, CUDA-compatible
PyTorch 2.7.1, JAX 0.5.3, and `transformers==4.53.2`. The upstream openpi
checkout and data are intentionally kept outside this small results repository
on the original server:

```bash
export OPENPI_DIR=/cfsdata/chenjinfeng/projects/openpi
export PERSONAL_DIR=/cfsdata/chenjinfeng/projects/openpi-libero-reproduction
export OPENPI_DATA_HOME=/cfsdata/chenjinfeng/openpi_cache
export HF_HOME=/cfsdata/chenjinfeng/hf_cache
export HF_LEROBOT_HOME=/cfsdata/chenjinfeng/datasets
export TMPDIR=/cfsdata/chenjinfeng/tmp
export MUJOCO_GL=egl
```

Initialize submodules in a fresh clone and install openpi/LIBERO following the
upstream instructions:

```bash
git clone --recurse-submodules https://github.com/Jinfeng50/openpi-libero-reproduction.git
cd openpi-libero-reproduction
git submodule update --init --recursive
git -C third_party/openpi submodule update --init --recursive
cd third_party/openpi
uv sync
uv pip install -e third_party/libero
```

For the existing server setup, `/cfsdata/chenjinfeng/projects/openpi` is the
openpi working tree and `/cfsdata/chenjinfeng/projects/LIBERO` is the simulator
checkout. See [`RUNBOOK.md`](RUNBOOK.md) for cache, weight, and environment
details.

### Official checkpoint sanity check

Start the upstream policy server with the official checkpoint, then run the
official client. The full 50-episode protocol is recorded in
[`docs/baseline.md`](docs/baseline.md).

```bash
cd "$OPENPI_DIR"
CUDA_VISIBLE_DEVICES=5 uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=/cfsdata/chenjinfeng/models/openpi/pi05_libero \

cd "$OPENPI_DIR/examples/libero"
uv run main.py --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 50 --args.host localhost --args.port 8000
```

### Full fine-tuning pipeline

The corrected pipeline resolves the actual openpi checkpoint layout and the
usual final step (`29999` for a 30,000-step run):

```bash
cd "$PERSONAL_DIR"
GPU_IDS=5,6 ./scripts/run_full_pipeline.sh pi05_libero_2gpu 2
```

Use `SKIP_NORM=1`, `SKIP_TRAIN=1`, or `SKIP_EVAL=1` to resume selected stages.
Every long stage writes a log under `experiments/<run>/logs/`.

## Innovation: DGTE

The openpi LIBERO client predicts an action chunk but executes only its first
five actions before asking the policy again. The next prediction therefore
overlaps the tail of the previous one. DGTE makes that overlap useful:

- continuous pose/action dimensions use freshness-weighted averaging;
- prediction disagreement increases the newest chunk's weight, improving
  response to a changed scene;
- the absolute gripper command is copied from the newest chunk, avoiding an
  invalid average between open and close;
- no model parameters, checkpoint files, or training data are changed.

Core implementation: [`src/openpi_libero_reproduction/temporal_ensemble.py`](src/openpi_libero_reproduction/temporal_ensemble.py).
The CPU-only unit tests are in [`tests/`](tests/).

Run a paired baseline/DGTE evaluation against an existing checkpoint:

```bash
cd "$PERSONAL_DIR"
EXP_NAME=pi05_libero_2gpu_20260610_1529 \
GPU_ID=5 \
N_EPISODES_PER_TASK=50 \
./scripts/run_temporal_ablation.sh
```

The script writes separate logs, rollout videos, and `sr_summary.csv` rows for
both controllers. On a node with eight free GPUs, the equivalent
`scripts/run_temporal_ablation_parallel.sh` launcher runs each independent
controller/suite pair on its own GPU and validates the same episode counts.

## Repository map

```text
docs/baseline.md                         official checkpoint protocol/result
docs/results.md                          completed experiment ledger
docs/method.md                           training and inference method
docs/innovation.md                       DGTE design and evaluation protocol
scripts/run_full_pipeline.sh             norm stats -> train -> serve -> eval
scripts/eval_libero_temporal.py          baseline/DGTE LIBERO client
scripts/run_temporal_ablation.sh         paired A/B runner
scripts/run_temporal_ablation_parallel.sh eight-GPU paired A/B runner
scripts/plot_results.py                   reproducible result figure
docs/world_model_direction.md             staged world-model + VLA plan
src/openpi_libero_reproduction/transition_dataset.py
                                         episode transition recorder/schema
src/openpi_libero_reproduction/world_model.py
                                         frozen-encoder latent critic
src/openpi_libero_reproduction/world_model_data.py
                                         episode-safe latent dataset/splits
scripts/precompute_world_latents.py      frozen visual latent extraction
scripts/train_world_model.py             Stage-A critic training
scripts/evaluate_world_model.py          offline critic metrics/calibration/latency
src/openpi_libero_reproduction/          tested, openpi-independent controller
experiments/                             CSV summaries and archived logs
third_party/openpi, third_party/LIBERO   pinned upstream submodules
```

## License and attribution

The experiment tooling in this repository is released under Apache-2.0. The
openpi and LIBERO submodules retain their upstream licenses. Follow the
upstream terms for pretrained weights and benchmark assets.
