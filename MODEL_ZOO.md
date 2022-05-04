# Model Zoo


|   Stage   | Model                | ARID Top-1 | Download   | Shell|
| --------- | -------------------- | --------------- | ---------- | ---- |
| Pre-train | UniFormer-B32-K600   | N/A               | [google](https://drive.google.com/file/d/1-DwdVf8w8lYj-iFpU40pfEpog9VE5PQB/view?usp=sharing) | N/A |
| Pre-train | UniFormer-B32-SSV2   | N/A               | [google](https://drive.google.com/file/d/1-rpMARXnyvyj6YUJkIvVqtna86egpjoS/view?usp=sharingg) | N/A |
| Pre-train | MViT-B32-K600        | N/A               | [google](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvit/k600.pyth) | N/A |
| Pre-train | SlowFast-R101-K700   | N/A               | [google](https://drive.google.com/file/d/1v1FdPUXBNRj-oKfctScT4L4qk8L1k3Gg/view?usp=sharing) | N/A |
| Adapt BN  | UniFormer-B32-K600   | 62.64               | [google](https://drive.google.com/file/d/1ST1TDcby8WioF7A3jZ6ZbA5XfZW-Mfdp/view?usp=sharing) | [run.sh](./exp_adapt_bn/uniformer_b32_k600/) |
| Adapt BN  | UniFormer-B32-SSV2   | 58.90               | [google](https://drive.google.com/file/d/1iFhi-LpXBnGuDjz6bbA6ceBYgaY8GCQX/view?usp=sharing) | [run.sh](./exp_adapt_bn/uniformer_b32_ssv2/) |
| Adapt BN  | MViT-B32-K600        | 58.14               | [google](https://drive.google.com/file/d/1myUhgFEZUnCoXqOnh1ASIFG2RANW2epx/view?usp=sharing) | [run.sh](./exp_adapt_bn/mvit_b32_k600/) |
| Adapt BN  | SlowFast-R101-K700   | 57.79               | [google](https://drive.google.com/file/d/1bzdgkHhbXtVySc7VOgB96L_7AgeuiaCc/view?usp=sharing) | [run.sh](./exp_adapt_bn/mvit_b32_k600/) |
| Pseudo1   | UniFormer-B32-K600   | 83.50               | [google](https://drive.google.com/file/d/1WlBNtsY0NTQdBlUslWYyopfsoEsNK-E4/view?usp=sharing) | [run.sh](./exp_pseudo_stage1/uniformer_b32_k600/) |
| Pseudo1   | UniFormer-B32-SSV2   | 81.04               | [google](https://drive.google.com/file/d/1mO6zbU1GYQYdQjXVQazXpdqMugXGVyIC/view?usp=sharing) | [run.sh](./exp_pseudo_stage1/uniformer_b32_ssv2/) |
| Pseudo1   | MViT-B32-K600        | 81.68              | [google](https://drive.google.com/file/d/1M3VYGPDQ6twBW_aWA17wq3kLhffYpNkQ/view?usp=sharing) | [run.sh](./exp_pseudo_stage1/mvit_b32_k600/) |
| Pseudo1   | SlowFast-R101-K700   | 80.78               | [google](https://drive.google.com/file/d/1_-2sJD49V9rv74ghAXme09yEERU6acJo/view?usp=sharing) | [run.sh](./exp_pseudo_stage1/sf32_k700/) |
| Pseudo2   | UniFormer-B32-K600   | 87.84               | [google](https://drive.google.com/file/d/1UAxKgr9z_eKaxDVrPM2X3mxMZl8MmCpg/view?usp=sharing) | [run.sh](./exp_pseudo_stage2/uniformer_b32_k600/) |
| Pseudo2   | UniFormer-B32-SSV2   | 85.95               | [google](https://drive.google.com/file/d/18rWv9UvnRReX1W3Dr7YKSRLi1YtOJRXh/view?usp=sharing) | [run.sh](./exp_pseudo_stage2/uniformer_b32_ssv2/) |
| Pseudo2   | MViT-B32-K600        | 86.63              | [google](https://drive.google.com/file/d/1Zq-ZDjMMggZJUUFKMZ62ts4FIS2B1ZjZ/view?usp=sharing) | [run.sh](./exp_pseudo_stage2/mvit_b32_k600/) |
| Pseudo2   | SlowFast-R101-K700   | 85.63               | [google](https://drive.google.com/file/d/1zQbhPGr7vw4zkgjTJGbbr3HEc4p7CElc/view?usp=sharing) | [run.sh](./exp_pseudo_stage2/sf32_k700/) |
| Pseudo3   | UniFormer-B32-K600   | 89.48               | [google](https://drive.google.com/file/d/1wvwYj0ZbyfDvmFn5eOwK2UdbxCBeMGWY/view?usp=sharing) | [run.sh](./exp_pseudo_stage3/uniformer_b32_k600/) |
| Pseudo3   | UniFormer-B32-SSV2   | 88.74               | [google](https://drive.google.com/file/d/1EIVUbQCS-HQX-SEn-D41cF6hPM18GOR7/view?usp=sharing) | [run.sh](./exp_pseudo_stage3/uniformer_b32_ssv2/) |
| Pseudo3   | MViT-B32-K600        | 88.75               | [google](https://drive.google.com/file/d/1Dr3Hdqn4oGX478M1RY1jCIzE5_S9Q4rW/view?usp=sharing) | [run.sh](./exp_pseudo_stage3/mvit_b32_k600/) |
| Pseudo3   | SlowFast-R101-K700   | 88.59               | [google](https://drive.google.com/file/d/1jcpbW_l-Vc3ZyzksEQ8xQVao56N-l6Dx/view?usp=sharing) | [run.sh](./exp_pseudo_stage3/sf32_k700/) |
| Pseudo4   | UniFormer-B32-K600   | 89.91               | [google](https://drive.google.com/file/d/1MshRoDb0YXIfmhNEhfcYsC7708x-zAJC/view?usp=sharing) | [run.sh](./exp_pseudo_stage4/uniformer_b32_k600/) |
| Pseudo4   | UniFormer-B32-SSV2   | 90.25               | [google](https://drive.google.com/file/d/1z9JYp1uAVypfK5BfE9ZAvNU9HSHbxvOD/view?usp=sharing) | [run.sh](./exp_pseudo_stage4/uniformer_b32_ssv2/) |
| Pseudo4   | MViT-B32-K600        | 90.30               | [google](https://drive.google.com/file/d/1GGb-KtcTi06rIuFyHT0Yrk99Dn32UpHF/view?usp=sharing) | [run.sh](./exp_pseudo_stage4/mvit_b32_k600/) |
| Pseudo4   | SlowFast-R101-K700   | 89.49               | [google](https://drive.google.com/file/d/1MshRoDb0YXIfmhNEhfcYsC7708x-zAJC/view?usp=sharing) | [run.sh](./exp_pseudo_stage4/sf32_k700/) |
| Pseudo4   | UniFormer-B32†-SSV2   | 89.51               | [google](https://drive.google.com/file/d/1yka9cF4rBHT5lTZRRHfRf455oiB6zaaO/view?usp=sharing) | [run.sh](./exp_experts/uniformer_b32_ssv2/) |

**Note:**
1. All models are trained with 32 frames that are uniformly sampled from the raw videos by default, except that the UniFormer-B32† is trained with dense sampling.
2. We used all the videos in [ARID](https://xuyu0010.github.io/arid.html) (a total of 6207 videos) for validation. For training, we generate pseudo labels for these videos. 

You can reuse all these models via setting `TRAIN.CHECKPOINT_FILE_PATH` and `TEST.CHECKPOINT_FILE_PATH`.
