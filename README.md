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



### Inference and Post-Processing

After pseudo labeling, we will get five well-trained models in five experiment(exp) folders as the followings, and the weight will be saved in `dark/best.pyth` under every exp folder. 

* [exp_experts/uniformer_b32_ssv2_ce](exp_experts/uniformer_b32_ssv2_ce)
* [exp_pseudo_arid_stage4/mvit_b32_k600_dp0.3_ce](exp_pseudo_arid_stage4/mvit_b32_k600_dp0.3_ce)
* [exp_pseudo_arid_stage4/sf32_k700_ce](exp_pseudo_arid_stage4/sf32_k700_ce)
* [exp_pseudo_arid_stage4/uniformer_b32_k600_ce](exp_pseudo_arid_stage4/uniformer_b32_k600_ce)
* [exp_pseudo_arid_stage4/uniformer_b32_ssv2_ce](exp_pseudo_arid_stage4/uniformer_b32_ssv2_ce). 



#### Inference with Well-Trained Models 
We use two types of TTA in inference. First we use multi-crop and multi-view for each fragment, then we use gamma correction to enhance the videos. Gamma correction is integrated into the inference process.  We provide the script `test.sh` under every exp folder to infer the model. You can easily run the script like 
```bash
bash exp_pseudo_arid_stage4/uniformer_b32_ssv2_ce/test.sh
```


**[Note]:**
We need to update `test.sh`:
```yaml
TEST.DATA_SELECT: test #the set name to infer, using `test` as the final test set .
TEST.NUM_ENSEMBLE_VIEWS: 3 #the view number for every fragment, and default `3` works best.
TEST.NUM_SPATIAL_CROPS: 3 #the crop number for every fragment, and default `3` works best.
TEST.CHECKPOINT_FILE_PATH: exp_folder/dark/best.pyth #the weight path of best model, such as `exp_folder/dark/best.pyth`
```

>We generated two types of results:1(view)x3(crop) and 3(view)x3(crop) under every exp folder at the same time to facilitate voting in post-processing, which are a total of 2(type)*5(model) results.


#### Post-Processing and Voting
We have constructed five different ensemble methods and used voting to achieve the best performance. We use `mode` to represent different ensemble methods in `test_ensemble_vote.py`. There are three core functions:

* `select_thres`: Complete a certain kind of ensemble method.
* `generate_final_sub`: Generate a reslut csv file for one ensemble method.
* `vote_for_sub`: Vote multiple results to generate a new result.

> It should be noted that we need to update the paths of the previous 2(type)*5(model) results in the select_thres function.




## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.


## Acknowledgement

This repository is built based on [UniFormer](https://github.com/Sense-X/UniFormer/tree/main/video_classification) and [SlowFast](https://github.com/facebookresearch/SlowFast) repository.