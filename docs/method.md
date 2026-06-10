# Method

## Stage 1: Base Reproduction (current)
- Backbone: PaliGemma 3B (frozen during loading, all params updated during FT)
- Action expert: π0.5 flow matching head
- Training: 30k steps, batch=64, FSDP over 2×A800, lr=2.5e-5, AdamW, bf16

## Stage 2: Multimodal Fusion (planned, TBD)
- Direction TBD pending Stage 1 results
- Candidates: PointVLA late fusion / DepthVLA MoT / radar-VLA
