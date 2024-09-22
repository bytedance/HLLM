#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# Use 8GPUs for batch_size = 8x64 = 512
# fixed temperature performs better
cd code && python3 main.py \
--config_file IDNet/hstu.yaml overall/ID.yaml \
--optim_args.learning_rate 1e-3 \
--loss nce \
--train_batch_size 16 \
--MAX_ITEM_LIST_LENGTH 55 \
--epochs 201 \
--dataset amazon_books \
--hidden_dropout_prob 0.5 \
--attn_dropout_prob 0.5 \
--n_layers 16 \
--n_heads 8 \
--item_embedding_size 64 \
--hstu_embedding_size 64 \
--fix_temp True \
--data_split False \
--show_progress True \
--update_interval 100 \
--optim_args.weight_decay 0.0 \
--seed 42 \
--stopping_step 10 