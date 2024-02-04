#!/bin/bash

# mod ucaps
export CUDA_VISIBLE_DEVICES=0

python evaluate.py --root_dir /path/to/dataset/ \
                    --gpus 1 \
                    --save_image 1 \
                    --model_name effiViTcaps \
                    --dataset iseg2017 \
                    --checkpoint_path path/to/checkpoint.ckpt \
                    --val_patch_size 32 32 32 \
                    --sw_batch_size 16 \
                    --overlap 0.75 \
                    --output_dir path/to/save_dir/of/predicted_images/