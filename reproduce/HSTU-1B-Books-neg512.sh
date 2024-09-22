#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# batch_size = 16GPUs * 8 = 128
cd code && python3 main.py \
--config_file IDNet/hstu.yaml overall/ID_deepspeed.yaml \
--optim_args.learning_rate 1e-3 \
--loss nce \
--train_batch_size 8 \
--MAX_ITEM_LIST_LENGTH 50 \
--epochs 201 \
--dataset amazon_books \
--hidden_dropout_prob 0.5 \
--attn_dropout_prob 0.5 \
--n_layers 22 \
--n_heads 32 \
--item_embedding_size 2048 \
--hstu_embedding_size 2048 \
--fix_temp True \
--num_negatives 512 \
--show_progress True \
--update_interval 100 \
--checkpoint_dir checkpoint_dir \
--stopping_step 10 