# [HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling](https://arxiv.org/abs/2409.12740)

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2409.12740-da282a.svg)](https://arxiv.org/abs/2409.12740)
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-ByteDance/HLLM-yellow)](https://huggingface.co/ByteDance/HLLM)
[![Recommendation](https://img.shields.io/badge/Task-Recommendation-blue)]()

</div>

## 🔥 Update
- [2024.09.20] Codes and Weights are released !


## Installation

1. Install packages via `pip3 install -r requirements.txt`. 
Some basic packages are shown below :
```
pytorch==2.1.0
deepspeed==0.14.2
transformers==4.41.1
lightning==2.4.0
flash-attn==2.5.9post1
fbgemm-gpu==0.5.0 [optional for HSTU]
sentencepiece==0.2.0 [optional for Baichuan2]
```
2. Prepare `PixelRec` and `Amazon Book Reviews` Datasets:
    1. Download `PixelRec` Interactions and Item Information from [PixelRec](https://github.com/westlake-repl/PixelRec) and put into the dataset and information folder.
    2. Download `Amazon Book Reviews` [Interactions](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv) and [Item Information](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz), process them by `process_books.py`, and put into the dataset and information folder. We also provide [Interactions](https://huggingface.co/ByteDance/HLLM/resolve/main/Interactions/amazon_books.csv) and [Item Information](https://huggingface.co/ByteDance/HLLM/resolve/main/ItemInformation/amazon_books.csv) of Books after processing.
    3. Please note that Interactions and Item Information should be put into two folders like:
        ```bash
        ├── dataset # Store Interactions
        │   ├── amazon_books.csv
        │   ├── Pixel1M.csv
        │   ├── Pixel200K.csv
        │   └── Pixel8M.csv
        └── information # Store Item Information
            ├── amazon_books.csv
            ├── Pixel1M.csv
            ├── Pixel200K.csv
            └── Pixel8M.csv
        ``` 
        Here dataset represents **data_path**, and infomation represents **text_path**.
3. Prepare pre-trained LLM models, such as [TinyLlama](https://github.com/jzhang38/TinyLlama), [Baichuan2](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base).

## Training
To train HLLM on PixelRec / Amazon Book Reviews, you can run the following command.

> Set `master_addr`, `master_port`, `nproc_per_node`, `nnodes` and `node_rank` in environment variables for multinodes training.

> All hyper-parameters (except model's config) can be found in code/REC/utils/argument_list.py and passed through CLI. More model's hyper-parameters are in `IDNet/*` or `HLLM/*`. 

```python
# Item and User LLM are initialized by specific pretrain_dir.
python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \ # We use deepspeed for training by default.
--loss nce \
--epochs 5 \
--dataset {Pixel200K / Pixel1M / Pixel8M / amazon_books} \
--train_batch_size 16 \
--MAX_TEXT_LENGTH 256 \
--MAX_ITEM_LIST_LENGTH 10 \
--checkpoint_dir saved_path \
--optim_args.learning_rate 1e-4 \
--item_pretrain_dir item_pretrain_dir \ # Set to LLM dir.
--user_pretrain_dir user_pretrain_dir \ # Set to LLM dir.
--text_path text_path \ # Use absolute path to text files.
--text_keys '[\"title\", \"tag\", \"description\"]' # Please remove tag in books dataset.
```
> You can use `--gradient_checkpointing True` and `--stage 3` with deepspeed to save memory.

You can also train ID-based models by the following command.
```python
python3 main.py \
--config_file overall/ID.yaml IDNet/{hstu / sasrec / llama_id}.yaml \
--loss nce \
--epochs 201 \
--dataset {Pixel200K / Pixel1M / Pixel8M / amazon_books} \
--train_batch_size 64 \
--MAX_ITEM_LIST_LENGTH 10 \
--optim_args.learning_rate 1e-4
```


To reproduce our experiments on Pixel8M and Books you can run scripts in `reproduce` folder. You should be able to reproduce the following results.
> For ID-based models, we follow the hyper-parameters from [PixelRec](https://github.com/westlake-repl/PixelRec) and [HSTU](https://github.com/facebookresearch/generative-recommenders/tree/main).

| Method        | Dataset | Negatives | R@10       | R@50      | R@200     | N@10      | N@50      | N@200     |
| ------------- | ------- |---------- | ---------- | --------- |---------- | --------- | --------- | --------- |
| HSTU          | Pixel8M | 5632      | 4.83       | 10.30     | 18.28     | 2.75      | 3.94      | 5.13      |
| SASRec        | Pixel8M | 5632      | 5.08       | 10.62     | 18.64     | 2.92      | 4.12      | 5.32      |
| HLLM-1B       | Pixel8M | 5632      | **6.13**   | **12.48** | **21.18** | **3.54**  | **4.92**  | **6.22**  |
| HSTU-large    | Books   | 512       | 5.00       | 11.29     | 20.13     | 2.78      | 4.14      | 5.47      |
| SASRec        | Books   | 512       | 5.35       | 11.91     | 21.02     | 2.98      | 4.40      | 5.76      |
| HLLM-1B       | Books   | 512       | **6.97**   | **14.61** | **24.78** | **3.98**  | **5.64**  | **7.16**  |
| HSTU-large    | Books   | 28672     | 6.50       | 12.22     | 19.93     | 4.04      | 5.28      | 6.44      |
| HLLM-1B       | Books   | 28672     | 9.28       | 17.34     | 27.22     | 5.65      | 7.41      | 8.89      |
| HLLM-7B       | Books   | 28672     | **9.39**   | **17.65** | **27.59** | **5.69**  | **7.50**  | **8.99**  |

## Inference
We provide fine-tuned HLLM models for evaluation, you can download from the following links or hugginface. Remember put the weights to `checkpoint_dir`.

| Model | Dataset | Weights |
|:---|:---|:---|
|HLLM-1B | Pixel8M | [HLLM-1B-Pixel8M](https://huggingface.co/ByteDance/HLLM/resolve/main/1B_Pixel8M/pytorch_model.bin)
|HLLM-1B | Books | [HLLM-1B-Books-neg512](https://huggingface.co/ByteDance/HLLM/resolve/main/1B_books_neg512/pytorch_model.bin)
|HLLM-1B | Books | [HLLM-1B-Books](https://huggingface.co/ByteDance/HLLM/resolve/main/1B_books/pytorch_model.bin)
|HLLM-7B | Books | [HLLM-7B-Books](https://huggingface.co/ByteDance/HLLM/resolve/main/7B_books/pytorch_model.bin)

> Please ensure compliance with the respective licenses of [TinyLlama-1.1B](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) and [Baichuan2-7B](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/blob/main/Community%20License%20for%20Baichuan%202%20Model.pdf) when using corresponding weights.

Then you can evaluate models by the following command (the same as training but val_only).
```python
python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \ # We use deepspeed for training by default.
--loss nce \
--epochs 5 \
--dataset {Pixel200K / Pixel1M / Pixel8M / amazon_books} \
--train_batch_size 16 \
--MAX_TEXT_LENGTH 256 \
--MAX_ITEM_LIST_LENGTH 10 \
--checkpoint_dir saved_path \
--optim_args.learning_rate 1e-4 \
--item_pretrain_dir item_pretrain_dir \ # Set to LLM dir.
--user_pretrain_dir user_pretrain_dir \ # Set to LLM dir.
--text_path text_path \ # Use absolute path to text files.
--text_keys '[\"title\", \"tag\", \"description\"]' \ # Please remove tag in books dataset.
--val_only True # Add this for evaluation
```



## Citation

If our work has been of assistance to your work, feel free to give us a star ⭐ or cite us using :  

```
@article{HLLM,
title={HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling},
author={Junyi Chen and Lu Chi and Bingyue Peng and Zehuan Yuan},
journal={arXiv preprint arXiv:2409.12740},
year={2024}
}
```

> Thanks to the excellent code repository [RecBole](https://github.com/RUCAIBox/RecBole), [VisRec](https://github.com/ialab-puc/VisualRecSys-Tutorial-IUI2021), [PixelRec](https://github.com/westlake-repl/PixelRec) and [HSTU](https://github.com/facebookresearch/generative-recommenders/tree/main) ! 
> HLLM is released under the Apache License 2.0, some codes are modified from HSTU and PixelRec, which are released under the Apache License 2.0 and MIT License, respectively.