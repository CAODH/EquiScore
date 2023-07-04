#!/bin/bash 
# copyright by caoduanhua(caodh@zju.edu.cn)

source /home/caoduanhua/anaconda3/bin/activate prolif
# export TORCH_DISTRIBUTED_DEBUG=INFO

export CUDA_VISIBLE_DEVICES="0,1,2,3"
# for multi screening task ,you can do cycle for this command
cd /home/caoduanhua/score_function/GNN/EquiScore
command=`python  Screening.py \
--ngpu 1 \
--test \
--test_path data/sample_data/ \
--test_name tmp_pockets \
--pred_save_path  data/test_results/EquiScore_tmp_pockets_1.pkl`
state=$command