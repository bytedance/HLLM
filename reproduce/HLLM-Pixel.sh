#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# 1B: 32 A100s for ≈ 4.1days
cd code && python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--MAX_ITEM_LIST_LENGTH 10 --epochs 5 --optim_args.learning_rate 1e-4 \
--checkpoint_dir saved_dir \
--loss nce --MAX_TEXT_LENGTH 256 --dataset Pixel8M \
--text_path text_path \
--item_pretrain_dir item_pretrain_dir \
--user_pretrain_dir user_pretrain_dir \
--train_batch_size 16