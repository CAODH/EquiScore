#!/bin/bash 
# copyright by caoduanhua(caodh@zju.edu.cn)

source ~/anaconda3/bin/activate EquiScore
# export TORCH_DISTRIBUTED_DEBUG=INFO

export CUDA_VISIBLE_DEVICES="0,1,2,3"
# for multi screening task ,you can do cycle for this command
cd ~/EquiScore
command=`python ./get_pocket/get_pocket.py \
--single_sdf_save_path ./data/sample_data/tmp_sdfs \
--recptor_pdb ./data/sample_data/sample_protein.pdb \
--docking_result ./data/sample_data/sample_compounds.sdf \
--pocket_save_dir ./data/sample_data/tmp_pockets \
--process_num 10`
state=$command
# rm tmp files
# rm -rf ./*.pdb
