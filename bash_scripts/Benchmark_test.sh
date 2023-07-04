#!/bin/bash 
# copyright by caoduanhua(caodh@zju.edu.cn)

source /home/caoduanhua/anaconda3/bin/activate prolif
# export TORCH_DISTRIBUTED_DEBUG=INFO

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# for multi screening task ,you can do cycle for this command
cd /home/caoduanhua/score_function/GNN/EquiScore

command=`python  Independent_test.py \
--ngpu 7 \
--MASTER_PORT 29501 \
--test \
--test_path /home/caoduanhua/score_function/data/independent_test/RTMScore_test_data/ \
--test_name dekois2_pocket_10_SP \
--test_flag test_dekois_20230704 \
--test_mode multi_pose`
state=$command