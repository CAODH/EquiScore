#!/bin/bash 
# copyright by caoduanhua(caodh@zju.edu.cn)

source /home/caoduanhua/anaconda3/bin/activate prolif
# export TORCH_DISTRIBUTED_DEBUG=INFO

export CUDA_VISIBLE_DEVICES="2,3"
# for multi screening task ,you can do cycle for this command
cd /home/caoduanhua/score_function/GNN/EquiScore
command=`python  ./get_pocket/get_pocket.py \
--single_sdf_save_path /home/house5/caoduanhua/dockingProjects/cache4challenge/EF_test/deepcoys_decoys/single_sdfs \
--recptor_pdb /home/caoduanhua/score_function/data/Screens/cache4challenge/CBLB_8GCY_SPLIT_FROM_DOCK_RESULT.pdb \
--docking_result /home/house5/caoduanhua/dockingProjects/cache4challenge/enamine/glide-dock_SP_CBLB_8GCY_HST_pv.maegz \
--pocket_save_dir /home/house5/caoduanhua/dockingProjects/cache4challenge/EF_test/deepcoys_decoys/single_sdfs_pockets \
--process_num 20 \
--save_single_sdf`
state=$command
# rm tmp files
rm -rf ./*.pdb