#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py --log_dir path/to/output_dir/ \
                --gpus 1 \
                --accelerator ddp \
                --check_val_every_n_epoch 1000 \
                --max_epochs 40000 \
                --dataset iseg2017 \
                --model_name effiViTcaps \
                --root_dir path/to/dataset \
                --cache_rate 1.0 \
                --train_patch_size 32 32 32 \
                --num_workers 2 \
                --batch_size 2 \
                --num_samples 1 \
                --in_channels 2 \
                --out_channels 4 \
                --val_patch_size 32 32 32 \
                --val_frequency 1000 \
                --sw_batch_size 16 \
                --overlap 0.75