# Model Zoo


|   Stage   | Model                | ARID Top-1 | Download   | Shell|
| --------- | -------------------- | --------------- | ---------- | ---- |
| Pre-train | UniFormer-B32-K600   | -               | [google](https://drive.google.com/file/d/1-DwdVf8w8lYj-iFpU40pfEpog9VE5PQB/view?usp=sharing) | - |
| Pre-train | UniFormer-B32-SSV2   | -               | [google](https://drive.google.com/file/d/1-rpMARXnyvyj6YUJkIvVqtna86egpjoS/view?usp=sharingg) | - |
| Pre-train | MViT-B32-K600        | -               | [google](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvit/k600.pyth) | - |
| Pre-train | SlowFast-R101-K700   | -               | [google](https://drive.google.com/file/d/1v1FdPUXBNRj-oKfctScT4L4qk8L1k3Gg/view?usp=sharing) | - |
| Adapt BN  | UniFormer-B32-K600   | -               | [google]() | [run.sh]() |
| Adapt BN  | UniFormer-B32-SSV2   | -               | [google]() | [run.sh]() |
| Adapt BN  | MViT-B32-K600        | -               | [google]() | [run.sh]() |
| Adapt BN  | SlowFast-R101-K700   | -               | [google]() | [run.sh]() |
| Pseudo1   | UniFormer-B32-K600   | -               | [google]() | [run.sh]() |
| Pseudo1   | UniFormer-B32-SSV2   | -               | [google]() | [run.sh]() |
| Pseudo1   | MViT-B32-K600        | -               | [google]() | [run.sh]() |
| Pseudo1   | SlowFast-R101-K700   | -               | [google]() | [run.sh]() |
| Pseudo2   | UniFormer-B32-K600   | -               | [google]() | [run.sh]() |
| Pseudo2   | UniFormer-B32-SSV2   | -               | [google]() | [run.sh]() |
| Pseudo2   | MViT-B32-K600        | -               | [google]() | [run.sh]() |
| Pseudo2   | SlowFast-R101-K700   | -               | [google]() | [run.sh]() |
| Pseudo3   | UniFormer-B32-K600   | -               | [google]() | [run.sh]() |
| Pseudo3   | UniFormer-B32-SSV2   | -               | [google]() | [run.sh]() |
| Pseudo3   | MViT-B32-K600        | -               | [google]() | [run.sh]() |
| Pseudo3   | SlowFast-R101-K700   | -               | [google]() | [run.sh]() |
| Pseudo4   | UniFormer-B32-K600   | -               | [google]() | [run.sh]() |
| Pseudo4   | UniFormer-B32-SSV2   | -               | [google]() | [run.sh]() |
| Pseudo4   | MViT-B32-K600        | -               | [google]() | [run.sh]() |
| Pseudo4   | SlowFast-R101-K700   | -               | [google]() | [run.sh]() |
| Pseudo4   | UniFormer-B32†-SSV2   | -               | [google]() | [run.sh]() |

**Note:**
1. All models are trained with 32 frames that uniformly sampled from the raw videos by default, except that the UniFormer-B32† is trained with dense sampling.
2. We used all the videos in [ARID](https://xuyu0010.github.io/arid.html) (totally 6207 videos) for validation. For training, we generate pseudo labels for these videos. 

You can reuse all these models via setting `TRAIN.CHECKPOINT_FILE_PATH` and `TEST.CHECKPOINT_FILE_PATH`.