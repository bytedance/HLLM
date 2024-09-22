#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# 1B: 128 A100s for ≈ 0.5days
# 7B: 128 A100s for ≈ 2days
# For Books, training with a sequence length of 55 and 56*512=28672 negatives in total, serving in a sequence length of 50.
cd code && python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--MAX_ITEM_LIST_LENGTH 55 \
--epochs 5 \
--optim_args.learning_rate 1e-4 \
--checkpoint_dir saved_dir \
--loss nce \
--MAX_TEXT_LENGTH 256 \
--scheduler_args.warmup 0.15 \
--dataset amazon_books \
--gradient_checkpointing True \
--text_keys '[\"title\",\"description\"]' \
--text_path text_path \
--item_pretrain_dir item_pretrain_dir \
--user_pretrain_dir user_pretrain_dir \
--train_batch_size 4 \
--data_split False \
--MAX_ITEM_LIST_LENGTH_TEST 50 \
--seed 42 \
--stage 3