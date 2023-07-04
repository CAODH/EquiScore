#!/bin/bash 
# copyright by caoduanhua(caodh@zju.edu.cn)

source /home/caoduanhua/anaconda3/bin/activate prolif
# export TORCH_DISTRIBUTED_DEBUG=INFO

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# for multi screening task ,you can do cycle for this command
cd /home/caoduanhua/score_function/GNN/EquiScore
command=`python  Screening.py \
--ngpu 2 \
--MASTER_PORT 29501 \
--test \
--test_path /home/house5/caoduanhua/dockingProjects/cache4challenge/enamine/ \
--test_name HTS_pockets_addition \
--pred_save_path  /home/house5/caoduanhua/dockingProjects/cache4challenge/enamine/EquiScore_results/HTS_pockets_addition.pkl`
state=$command