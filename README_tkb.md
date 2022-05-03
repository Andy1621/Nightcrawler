





# Infer and Post-Process

After pseudo labeling, we will get five well-trained models in five experiment(exp) folders: `exp_experts/uniformer_b32_ssv2_ce`,`exp_pseudo_arid_stage4/mvit_b32_k600_dp0.3_ce`,`exp_pseudo_arid_stage4/sf32_k700_ce`,`exp_pseudo_arid_stage4/uniformer_b32_k600_ce` and `exp_pseudo_arid_stage4/uniformer_b32_ssv2_ce`. The weight will be saved in `dark/best.pyth` under every exp folder.

## Infer with well-trained models 
We use two types of TTA in inference. First we use multi-crop and multi-view for each fragment, then we use gamma correction to enhance the videos.
We provide the script `test.sh` under every exp folder to infer the model. You can easily run the script through `bash test.sh`. Gamma correction is integrated in the inference process. Here are some parameters that need to be updated. 

* TEST.DATA_SELECT: the set name to infer, using `test` as the final test set .
* TEST.NUM_ENSEMBLE_VIEWS: the view number for every fragment, and default `3` works best.
* TEST.NUM_SPATIAL_CROPS: the crop number for every fragment, and default `3` works best.
* TEST.CHECKPOINT_FILE_PATH: the weight path of best model, such as `exp_folder/dark/best.pyth`

We generated two types of results:1(view)x3(crop) and 3(view)x3(crop) under every exp folder at the same time to facilitate voting in post-processing, which are a total of 2(type)*5(model) results.


## Post-process and vote
We have constructed five different ensemble methods and used voting to achieve the best performance. We use `mode` to represent different ensemble methods in `test_ensemble_vote.py`.
There are three core functions:

* `select_thres`: Complete a certain kind of ensemble method.
* `generate_final_sub`: Generate a reslut csv file for one ensemble method.
* `vote_for_sub`: Vote multiple results to generate a new result.

It should be noted that we need to update the paths of the previous 2(type)*5(model) results in the `select_thres` function.


