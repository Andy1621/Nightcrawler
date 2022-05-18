# Dataset

## Type
The [UG2+ 2022 Track2](https://codalab.lisn.upsaclay.fr/competitions/1112) provide 4 datasets:
1. Normal-light Training Dataset (2625 videos w/ label)
2. Normal-light Validation Dataset (330 videos w/ label)
3. Dark Training Dataset (3088 videos w/o label)
4. Dark Testing Dataset (3102 videos w/o label)

Besides, the organizers allow utilizing [ARID](https://xuyu0010.github.io/arid.html) dataset to validate and provide pseudo labels.

5. ARID Training Dataset (6207 videos w/o label).

## Prepartion
1. Download all the data in [UG2+ 2022 Track2](https://codalab.lisn.upsaclay.fr/competitions/1112) and [ARID_v1.5](https://xuyu0010.github.io/arid.html).
2. Prepare soft links in `./data`, we haved provided all the CSV files.
```shell
# soft link for UG2
ln -s your_data_path/dark-train dark_train
ln -s your_data_path/dry-run dry_run
ln -s your_data_path/labeled-train labeled_train
ln -s your_data_path/Test Test
ln -s your_data_path/ARID_v1.5/clips_v1.5 dark_train/Train
# copy data from ARID
cp -r your_data_path/ARID_v1.5/clips_v1.5/* ./dark_train/Train
```
After the above steps, you can simply set `DATA.PATH_PREFIX` as `data`.

## Usage

1. **[Supervised Training]**:
    - **Normal-light Training Dataset** and **Normal-light Validation Dataset** are used for supervised training, all the videos and labels are utilized to train the models.
    - **Dark Training Dataset** and **ARID Training Dataset** are used for **adapting BN**, only the videos are utilized to update the parameters in BN.
    - **ARID Training Dataset** is used for validation, all the videos and labels are utilized to select the best model.
2. **[Semi-supervised Training]**:
    - **Dark Training Dataset** and **ARID Training Dataset** are used for generating **pseudo labels**, only those pseudo labels with high confidence are utilized for training.
    - **Normal-light Training Dataset** and **Normal-light Validation Dataset** are also used for semi-supervised training.
    - **ARID Training Dataset** is still used for validation, all the videos and labels are utilized to select the best model.
3. **[Testing]**
    - **Dark Testing Dataset** is used for testing, only the videos are used for generating corresponding predictions.