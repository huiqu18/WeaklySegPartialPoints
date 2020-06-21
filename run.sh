#!/usr/bin/env bash


ratio='0.10'
dataset='MO'
for repeat in 1 2
do
# detection
cd ~/Research/TMI_weakly_seg/github/code_detection || exit
echo ${PWD}
python main.py --random-seed -1 --lr 0.0001 --batch-size 16 --epochs 80 \
  --gpus 0 --root-save-dir ../experiments/detection/${dataset}/${ratio}_repeat=${repeat}

# segmentation
cd ~/Research/TMI_weakly_seg/github/code_seg || exit
echo ${PWD}
python main.py --random-seed -1 --lr 0.0001 --batch-size 8 --epochs 100 \
  --gpus 0 --save-dir ../experiments/segmentation/${dataset}/${ratio}_repeat=${repeat} \
  --detection-results-dir ../experiments/detection/${dataset}/${ratio}_repeat=${repeat}/3/best/images_prob_maps
done