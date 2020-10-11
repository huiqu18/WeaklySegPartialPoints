#!/usr/bin/env bash

ratio='0.10'
dataset='MO'
repeat=1
python main.py --random-seed -1 --lr 0.0001 --batch-size 8 --epochs 100 \
  --gpus 0 --save-dir ../experiments/segmentation/${dataset}/${ratio}_repeat=${repeat} \
  --detection-results-dir ../experiments/detection/${dataset}/${ratio}_repeat=${repeat}/3/best/images_prob_maps

#dataset='MO'
#for repeat in 1 2 3 4 5
#do
#python test.py --model-path ../experiments/segmentation/${dataset}/0.25_repeat=${repeat}/checkpoints/checkpoint_best.pth.tar \
#  --save-dir ../experiments/segmentation/${dataset}/0.25/best
#done

