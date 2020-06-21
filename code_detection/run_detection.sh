#!/usr/bin/env bash

ratio='0.10'
dataset='MO'
repeat=1
# detection
python main.py --random-seed -1 --lr 0.0001 --batch-size 16 --epochs 80 \
  --gpus 0 --root-save-dir ../experiments/detection/${dataset}/${ratio}_repeat=${repeat}

#python test.py --img-dir ../data_for_train/LC/images/test --label-dir ../data/LC/labels_point \
#  --model-path ../experiments/detection/LC/ST-NU/3/checkpoints/checkpoint_best.pth.tar \
#  --threshold 0.40 --save-dir ../experiments/detection/LC/ST-NU/3/best