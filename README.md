# QANAS_alibaba

## Overview
**QANAS ~ QANAS12**: 12 experiments with different settings.

**pytorch.yml**: conda environment (same with the one posted to wechat last time)

**Datesets**: CIFAR-10 / CIFAR-100

**Two-step Training**: 1.search with ***train_search.py*** ; 2. train from scratch with ***train.py***.  (the intermedia results are saved at ***./ckpt/search/arch.pt*** which will be automatedly read by ***train.py***)

## Usage
Note that for QANAS ~ QANAS6, `path-to-dataset` is the path to CIFAR-100 dataset, while for QANAS7 ~ QANAS12, `path-to-dataset` is the path to CIFAR-10 dataset. One gpu is enough for each experiment and there's no need to change other settings (difference between settings are done in the code). The following two steps are suitable for all the totally 12 experiments.

Step 1. `CUDA_VISIBLE_DEVICES=0 python train_search.py --dataset_path path-to-dataset`

Step 2. `CUDA_VISIBLE_DEVICES=0 python train.py --dataset_path path-to-dataset`
