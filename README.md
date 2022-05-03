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

The training process is divided into five stages, including a basic model training and four stages of pseudo-label training iteration.

In each stage, we trained four methods, including K600 pre-trained MVIT, K600 pre-trained SlowFast, K600 pre-trained Uniformer,Something-Somethingv2 pre-trained Uniformer.

Each method at any stage has a folder where the config, training/test script are stored.

The arguments of each training/test script are mostly generic, and the training/test script does not need to be modified if you follow our path of datasets and training configs strictly.

If you modify weight path or data path, You may need to modify the following arguments.
```
DATA.PATH_PREFIX path_to_your_data \ *data* folder path
NUM_GPUS gpu_num \ 
TRAIN.CHECKPOINT_FILE_PATH pretrain_model_path \  pretrain model path in training
TEST.CHECKPOINT_FILE_PATH \  Testing model path
```

**1. Basic model training**

**[Note]:** 
Change the `TRAIN.CHECKPOINT_FILE_PATH` in `run.sh` with the MODEL_ZOO.md pretrain weights.

Simply run the training scripts in [exp_adapt_bn](exp_adapt_bn) as followed:
   ```shell
   bash ./exp_adapt_bn/xxxx/run.sh
   ```
And you will get the `best.pyth` weight model.


Change the arguments in `run.sh`,`TRAIN.CHECKPOINT_FILE_PATH` must be loaded with the MODEL_ZOO.md pretrain weights.

Simply run the testing scripts in [exp_adapt_bn](exp_adapt_bn) as followed:
   ```shell
   bash ./exp_adapt_bn/xxxx/test.sh
   ```
And you will get the test results of this stages and generate `.pkl` files to store in the folder.

`xxxx` is the different method. The above steps should be finished for four methods, respectively.

**2. Generate the pseudo label**

We provide a script to select the threshold in [gen_pesudo_tools](gen_pesudo_tools), the script will give you a sequence of threshold. 
You need to manually select the good threshold according to your demand. 
If you want to use this script, you need to change the pkl file path arguments in `generate_pseudo_label_arid_emu.py`.
```
python gen_pesudo_tools/generate_pseudo_label_arid_emu.py
```
We have given the set of thresholds we selected for each stage. You can run the following script to generate the pseudo-label files directly. X could be 1,2,3,4.
```
python gen_pesudo_tools/generate_pseudo_label_stageX.py
```
**3. Pseudo-label training iteration**

Simply run the training scripts in [exp_pseudo_arid_stageX] as followed:
   ```shell
   bash ./exp_adapt_bn/xxxx/run.sh
   ```
And you will get the `dark/best.pyth` weight model.

Simply run the testing scripts in [exp_pseudo_arid_stageX] as followed:
   ```shell
   bash ./exp_adapt_bn/xxxx/test.sh
   ```
And you will get the test results of this stages and generate `.pkl` files to store in the folder.

`xxxx` is the different method. The above steps should be finished for four methods, respectively.

**[Note]:** 
**2 and 3 are circularly executed during four iterations. The complete training process is as follows:**
   ```
   exp_adapt_bn --> generate_pseudo_label_stage1 --> exp_pseudo_arid_stage1
   --> generate_pseudo_label_stage2 --> exp_pseudo_arid_stage2
   --> generate_pseudo_label_stage3 --> exp_pseudo_arid_stage4
   --> generate_pseudo_label_stage4 --> exp_pseudo_arid_stage4
   ```
   
#### Tesing

#### Model Voting


#### Generating Pseudo Labels


#### Semi-upervised Training


