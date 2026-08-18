# Method

## Reproduction configuration

The completed fine-tuning run used the upstream `pi05_libero` configuration:

| Field | Value |
|---|---|
| Base checkpoint | `pi05_base` |
| Dataset | `physical-intelligence/libero` |
| Frames / episodes | 273,465 / 1,693 |
| Model | pi0.5 (PaliGemma + flow-matching action expert) |
| Trainable parameters | full model fine-tuning |
| Steps | 30,000 (final checkpoint directory: `29999`) |
| Global batch | 64 |
| Devices | 2x NVIDIA A800-SXM4-80GB, FSDP |
| Optimizer | AdamW, gradient clipping 1.0 |
| Learning-rate schedule | 10,000-step warmup, 5e-5 peak, cosine decay config |
| Precision | bfloat16 parameters / JAX mixed precision |
| Training wall time | approximately 28.4 hours, including final checkpoint save |

The training log is archived at
`experiments/pi05_libero_2gpu_20260610_1529/logs/02_train.log`. The run ended
with `loss=0.0110` and `grad_norm=0.0571` at step 29,900; those are diagnostics,
not benchmark success rates.

LIBERO normalization statistics were generated from the LIBERO training data
for this fine-tuning run. An official checkpoint evaluation must continue to use
the normalization statistics shipped with that checkpoint. Mixing the two
statistics files changes action de-normalization and invalidates a comparison.

## DGTE inference extension

`src/openpi_libero_reproduction/temporal_ensemble.py` implements a client-side
Disagreement-Gated Temporal Ensemble. Let `a_{k,t}` be the action for absolute
simulator time `t` predicted by chunk `k`. For continuous dimensions, DGTE uses

```text
w_k(t) = exp(-decay * age_k / replan_steps)
a_t    = sum_k w_k(t) a_{k,t} / sum_k w_k(t)
```

When the mean absolute disagreement from the newest chunk exceeds the
configured threshold, the newest weight is multiplied by
`1 + gate_strength * gate`, where `gate` is clipped to `[0, 1]`. The gripper
dimension is copied from the newest chunk because LIBERO treats it as an
absolute open/close command.

The method has no learned parameters and is therefore a clean inference-only
ablation. `baseline` and `dgte` are both supported by
`scripts/eval_libero_temporal.py`; `scripts/run_temporal_ablation.sh` runs them
against the same server and writes one CSV row per controller and suite.

## Planned comparison

DGTE results will only be added after the same checkpoint, seed, number of
episodes, and task suites have been run for both controllers. Until then the
result is intentionally marked `pending` in the README.
