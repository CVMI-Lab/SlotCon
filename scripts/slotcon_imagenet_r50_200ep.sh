#!/bin/bash

set -e
set -x

data_dir="./datasets/imagenet/"
output_dir="./output/slotcon_imagenet_r50_200ep"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port 12348 --nproc_per_node=8 \
    main_pretrain.py \
    --dataset ImageNet \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --arch resnet50 \
    --dim-hidden 4096 \
    --dim-out 256 \
    --num-prototypes 2048 \
    --teacher-momentum 0.99 \
    --teacher-temp 0.07 \
    --group-loss-weight 0.5 \
    \
    --batch-size 512 \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 200 \
    --fp16 \
    \
    --print-freq 10 \
    --save-freq 5 \
    --auto-resume \
    --num-workers 8
