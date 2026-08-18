# WandB 截图与实验留痕清单

本文档规定 2×A800 `pi05_libero` 全量微调实验需要从 WandB 保存哪些截图，以及这些截图分别用于证明什么。

## 目的

WandB 截图主要用于证明：

- 这次训练确实跑过；
- run 配置能被明确识别；
- 训练优化过程正常，没有明显崩溃；
- 两张 GPU 确实被使用；
- 后续评测结果能追溯到具体某一次训练 run。

注意：最终 LIBERO 成功率应以 eval log 和 `sr_summary.csv` 为准，不能只靠 WandB 截图汇报。

## 必须保存的截图

所有截图统一保存到：

```text
docs/figures/
```

推荐文件名：

| 文件名 | 截什么 | 用途 |
|---|---|---|
| `wandb_run_overview_2gpu.png` | 单个 run 的 overview 页面，包含 run name、config、step 数、主机、时间 | 证明具体是哪一次实验 |
| `train_loss_2gpu_30k.png` | `train/loss` 训练曲线，覆盖 30k steps | 展示 loss 是否正常下降 |
| `lr_schedule_2gpu_30k.png` | learning rate 曲线 | 验证 warmup / decay 是否正常 |
| `grad_norm_2gpu_30k.png` | gradient norm 曲线 | 判断梯度是否稳定，有无爆炸 |
| `throughput_2gpu.png` | step time、samples/sec、steps/sec 等吞吐指标 | 支撑 2×A800 训练效率说明 |
| `gpu_usage_2gpu.png` | 两张 GPU 的显存和利用率 | 证明两张 GPU 都实际参与训练 |

最低要求至少保存三张：

```text
docs/figures/wandb_run_overview_2gpu.png
docs/figures/train_loss_2gpu_30k.png
docs/figures/grad_norm_2gpu_30k.png
```

## 具体检查点

### Run Overview

截图里最好能看到：

- run name，例如 `pi05_libero_2gpu_YYYYMMDD_HHMM`；
- config name：`pi05_libero`；
- total steps：`30000`；
- batch size：`64`；
- 如果 WandB 有记录，最好也能看到 hostname 和 commit hash。

这张图用于证明实验身份，后面 README、release、博客里都可以引用。

### Train Loss

`train/loss` 应该整体下降。需要警惕：

- loss 变成 NaN；
- loss 从一开始就几乎不动；
- loss 突然爆炸且不恢复；
- run 在远小于 30k steps 的地方停止。

这张图是训练过程最核心的证据。

### Learning Rate

learning rate 曲线应该符合预期调度，通常能看到 warmup 和后续 decay。

这张图用于确认 scheduler 没配错，也能帮助判断断点续训是否正常。

### Grad Norm

grad norm 应该大体稳定。偶尔有尖峰可以接受，但持续爆炸不正常。

这张图用于证明训练没有明显梯度不稳定。

### Throughput

如果 WandB 有记录，保存以下任一类指标：

- step time；
- samples/sec；
- steps/sec；
- training throughput。

这张图用于说明 2×A800 的训练效率；本次完整训练日志记录的墙钟时间约为 28.4 小时（含最终 checkpoint 保存）。

### GPU Usage

如果 WandB 系统监控开启，保存：

- 两张 GPU 的显存占用；
- 两张 GPU 的 utilization；
- 如果有，也可以截 GPU power。

两张 GPU 都应该有明显显存占用；如果只有一张卡在工作，说明 FSDP / 可见卡配置可能有问题。

## README / Release 中怎么引用

WandB 截图用于证明训练过程：

```markdown
训练曲线和系统监控截图已保存到 `docs/figures/`。
```

最终 LIBERO SR 用 eval 日志证明：

```text
experiments/<EXP_NAME>/sr_summary.csv
```

不要只根据 WandB 截图报告 LIBERO 成功率。WandB 负责证明训练过程，`sr_summary.csv` 负责证明最终评测结果。
