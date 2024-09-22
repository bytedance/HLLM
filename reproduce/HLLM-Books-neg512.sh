#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# 1B: 16 H100s for ≈ 2days
cd code && python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--MAX_ITEM_LIST_LENGTH 50 \
--epochs 5 \
--optim_args.learning_rate 1e-4 \
--checkpoint_dir saved_dir \
--loss nce \
--MAX_TEXT_LENGTH 256 \
--dataset amazon_books \
--gradient_checkpointing True \
--text_keys '[\"title\",\"description\"]' \
--train_batch_size 8 \
--text_path text_path \
--item_pretrain_dir item_pretrain_dir \
--user_pretrain_dir user_pretrain_dir \
--num_negatives 512