#!/bin/bash 
# copyright by caoduanhua(caodh@zju.edu.cn)

source /home/caoduanhua/anaconda3/bin/activate prolif
# export TORCH_DISTRIBUTED_DEBUG=INFO

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# for multi screening task ,you can do cycle for this command
cd /home/caoduanhua/score_function/GNN/EquiScore
command=`python  Train.py \
--ngpu 1 \
--MASTER_PORT 29501 \
--train_keys /home/caoduanhua/score_function/GNN/config_keys_results/new_data_train_keys/train_keys/pdb_screen_neg_5_bind_keys_pose_enhanced_challenge_10/train_keys.pkl \
--val_keys /home/caoduanhua/score_function/GNN/config_keys_results/new_data_train_keys/train_keys/pdb_screen_neg_5_bind_keys_pose_enhanced_challenge_10/val_keys.pkl \
--test_keys /home/caoduanhua/score_function/GNN/config_keys_results/new_data_train_keys/train_keys/pdb_screen_neg_5_bind_keys_pose_enhanced_challenge_10/test_keys.pkl \
--lmdb_cache /home/caoduanhua/score_function/data/lmdbs/pose_challenge_cross_10`
state=$command