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


