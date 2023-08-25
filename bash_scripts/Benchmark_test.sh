#!/bin/bash 
# copyright by caoduanhua(caodh@zju.edu.cn)

source ~/anaconda3/bin/activate EquiScore
# export TORCH_DISTRIBUTED_DEBUG=INFO

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# for multi screening task ,you can do cycle for this command
cd ~/EquiScore

command=`python Independent_test.py \
--ngpu 1 \
--MASTER_PORT 29501 \
--test \
--test_path test_data_dir \
--test_name test_pocket_dir_name \
--test_flag flag_help_you_identity_dataset \
--test_mode multi_pose`
state=$command