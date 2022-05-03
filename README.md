# Nightcrawler
This repo is the top-2 solution for [UG2+ 2022 Track2](http://cvpr2022.ug2challenge.org/dataset22_t2.html).
We achieve 90.7% top-1 accuracy for [extreme dark video classification](https://codalab.lisn.upsaclay.fr/competitions/1112#results).


## Introduction
Our method consists of three steps:
1. **[Supervised Training]**: Only thosed videos under normal light are used for training. Those dark videos are used for adapting BN.
2. **[Model Voting]**: Different models are voted via specific threshold to generate more accurate pseudo labels.
3. **[Semi-supervised Training]**: Those darks videos with pseudo labels are used for training.

The final two steps are repeated four times, and we only select those pseudo with high confidence for training.


## Model Zoo
See [MODEL_ZOO.md](./MODEL_ZOO.md) for more details.

## Dataset
See [DATASET.md](./DATASET.md) for more details.

## Usage
### Installation

Please follow the installation instructions in [INSTALL.md](INSTALL.md). You may follow the instructions in [DATASET.md](DATASET.md) to prepare the datasets.

### Training

#### Supervised Training
> Fuckkkkkk

1. Download the pretrained models in our repository.

2. Simply run the training scripts in [exp_adapt_bn](exp_adapt_bn) as followed:
   ```shell
   bash ./exp/xxxx/run.sh
   ```
**[Note]:**

- set xxxx in `config.yaml`:
  ```yaml
  FUCK: True # fuck
  ```
- set xxxx in `run.sh`:
  ```shell
  TRAIN.CHECKPOINT_FILE_PATH: xxx # fuck
  ```

#### Tesing

#### Model Voting


#### Generating Pseudo Labels


#### Semi-upervised Training


