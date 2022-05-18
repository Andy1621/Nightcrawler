work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../slowfast \
python tools/run_net_bn_pseudo.py \
  --init_method tcp://localhost:10235 \
  --cfg $work_path/config.yaml \
  DATA.PATH_PREFIX ../../data \
  DATA.PATH_LABEL_SEPARATOR "," \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 5 \
  TRAIN.BATCH_SIZE 16 \
  NUM_GPUS 2 \
  SF.DROPOUT_RATE 0.3 \
  SOLVER.MAX_EPOCH 40 \
  SOLVER.BASE_LR 1e-4 \
  SOLVER.WARMUP_EPOCHS 10.0 \
  SOLVER.WEIGHT_DECAY 0.03 \
  DATA.TEST_CROP_SIZE 224 \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 1 \
  TEST.DATA_SELECT test \
  TEST.TEST_BEST True \
  TRAIN.ENABLE False \
  RNG_SEED 6666 \
  OUTPUT_DIR $work_path
