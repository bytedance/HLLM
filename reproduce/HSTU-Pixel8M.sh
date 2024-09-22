#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# Use 8GPUs for batch_size = 8x64 = 512
cd code && python3 main.py \
--config_file IDNet/hstu.yaml overall/ID.yaml \
--optim_args.learning_rate 1e-4 \
--loss nce \
--train_batch_size 64 \
--MAX_ITEM_LIST_LENGTH 10 \
--epochs 50 \
--dataset Pixel8M \
--stopping_step 5 \
--show_progress True \
--update_interval 100 