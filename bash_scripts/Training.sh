#!/bin/bash 
# copyright by caoduanhua(caodh@zju.edu.cn)

source ~/anaconda3/bin/activate EquiScore
# export TORCH_DISTRIBUTED_DEBUG=INFO

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# if you want to use lmdb_cache, you should set --test to replace --lmdb_cache lmdb_cache_dir
cd ~/EquiScore
command=`python Train.py \
--ngpu 1 \
--MASTER_PORT 29501 \
--train_keys ./data/data_splits/screen_model/train_keys.pkl \
--val_keys ./data/data_splits/screen_model/val_keys.pkl \
--test_keys ./data/data_splits/screen_model/test_keys.pkl \
--test`
state=$command