# Model Zoo


|   Stage   | Model                | ARID Top-1 | Download   | Shell|
| --------- | -------------------- | --------------- | ---------- | ---- |
| Pre-train | UniFormer-B32-K600   | -               | [google](https://drive.google.com/file/d/1-DwdVf8w8lYj-iFpU40pfEpog9VE5PQB/view?usp=sharing) | - |
| Pre-train | UniFormer-B32-SSV2   | -               | [google](https://drive.google.com/file/d/1-rpMARXnyvyj6YUJkIvVqtna86egpjoS/view?usp=sharingg) | - |
| Pre-train | MViT-B32-K600        | -               | [google](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvit/k600.pyth) | - |
| Pre-train | SlowFast-R101-K700   | -               | [google](https://drive.google.com/file/d/1v1FdPUXBNRj-oKfctScT4L4qk8L1k3Gg/view?usp=sharing) | - |
| Adapt BN  | UniFormer-B32-K600   | -               | [google]() | [run.sh](./exp_adapt_bn/uniformer_b32_k600/) |
| Adapt BN  | UniFormer-B32-SSV2   | -               | [google]() | [run.sh](./exp_adapt_bn/uniformer_b32_ssv2/) |
| Adapt BN  | MViT-B32-K600        | -               | [google]() | [run.sh](./exp_adapt_bn/mvit_b32_k600/) |
| Adapt BN  | SlowFast-R101-K700   | -               | [google]() | [run.sh](./exp_adapt_bn/mvit_b32_k600/) |
| Pseudo1   | UniFormer-B32-K600   | -               | [google]() | [run.sh](./exp_pseudo_stage1/uniformer_b32_k600/) |
| Pseudo1   | UniFormer-B32-SSV2   | -               | [google]() | [run.sh](./exp_pseudo_stage1/uniformer_b32_ssv2/) |
| Pseudo1   | MViT-B32-K600        | -               | [google]() | [run.sh](./exp_pseudo_stage1/mvit_b32_k600/) |
| Pseudo1   | SlowFast-R101-K700   | -               | [google]() | [run.sh](./exp_pseudo_stage1/sf32_k700/) |
| Pseudo2   | UniFormer-B32-K600   | -               | [google]() | [run.sh](./exp_pseudo_stage2/uniformer_b32_k600/) |
| Pseudo2   | UniFormer-B32-SSV2   | -               | [google]() | [run.sh](./exp_pseudo_stage2/uniformer_b32_ssv2/) |
| Pseudo2   | MViT-B32-K600        | -               | [google]() | [run.sh](./exp_pseudo_stage2/mvit_b32_k600/) |
| Pseudo2   | SlowFast-R101-K700   | -               | [google]() | [run.sh](./exp_pseudo_stage2/sf32_k700/) |
| Pseudo3   | UniFormer-B32-K600   | -               | [google]() | [run.sh](./exp_pseudo_stage3/uniformer_b32_k600/) |
| Pseudo3   | UniFormer-B32-SSV2   | -               | [google]() | [run.sh](./exp_pseudo_stage3/uniformer_b32_ssv2/) |
| Pseudo3   | MViT-B32-K600        | -               | [google]() | [run.sh](./exp_pseudo_stage3/mvit_b32_k600/) |
| Pseudo3   | SlowFast-R101-K700   | -               | [google]() | [run.sh](./exp_pseudo_stage3/sf32_k700/) |
| Pseudo4   | UniFormer-B32-K600   | -               | [google]() | [run.sh](./exp_pseudo_stage4/uniformer_b32_k600/) |
| Pseudo4   | UniFormer-B32-SSV2   | -               | [google]() | [run.sh](./exp_pseudo_stage4/uniformer_b32_ssv2/) |
| Pseudo4   | MViT-B32-K600        | -               | [google]() | [run.sh](./exp_pseudo_stage4/mvit_b32_k600/) |
| Pseudo4   | SlowFast-R101-K700   | -               | [google]() | [run.sh](./exp_pseudo_stage4/sf32_k700/) |
| Pseudo4   | UniFormer-B32†-SSV2   | -               | [google]() | [run.sh](./exp_experts/uniformer_b32_ssv2/) |

**Note:**
1. All models are trained with 32 frames that are uniformly sampled from the raw videos by default, except that the UniFormer-B32† is trained with dense sampling.
2. We used all the videos in [ARID](https://xuyu0010.github.io/arid.html) (a total of 6207 videos) for validation. For training, we generate pseudo labels for these videos. 

You can reuse all these models via setting `TRAIN.CHECKPOINT_FILE_PATH` and `TEST.CHECKPOINT_FILE_PATH`.