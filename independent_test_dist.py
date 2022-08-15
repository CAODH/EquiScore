import pickle
# import optuna
# from optuna.trial import TrialState

import time
import numpy as np
import utils
from utils import *
import torch.nn as nn
import torch
import time
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
# os.envirment[]
from collections import defaultdict
import argparse
import time
from torch.utils.data import DataLoader          
# from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from GTE_net import GTENet
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())                            
from graphformer_dataset import graphformerDataset, collate_fn, DTISampler
now = time.localtime()
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print (s)
os.chdir(os.path.abspath(os.path.dirname(__file__)))

from torch.multiprocessing import Process
def run(local_rank,args,*more_args,**kwargs):

    args.local_rank = local_rank
    torch.distributed.init_process_group(backend="nccl",init_method='env://',rank = args.local_rank,world_size = args.ngpu)  # 并行训练初始化，'nccl'模式
    torch.cuda.set_device(args.local_rank) 

    seed_torch(seed = args.seed + args.local_rank)

    args_dict = vars(args)
    if args.FP:
        args.N_atom_features = 39
    else:
        args.N_atom_features = 28

    model = GTENet(args) if args.gnn_model == 'graph_transformer_dgl' else None
   

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = args.local_rank
    best_name = args.save_model
    save_path = best_name.replace('/save_best_model.pt','')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.test_path = os.path.join(args.test_path,args.test_name)
    #model, args.device,args,args.save_model
    model = utils.initialize_model(model, args.device,args, load_save_file = best_name )[0]

    if args.loss_fn == 'bce_loss':
        loss_fn = nn.BCELoss().to(args.device)# 
    elif args.loss_fn == 'focal_loss':
        loss_fn = FocalLoss().to(args.device)
    elif args.loss_fn == 'cross_entry':
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smothing).to(args.device)
    elif args.loss_fn == 'mse_loss':
        loss_fn = nn.MSELoss().to(args.device)
    elif args.loss_fn == 'poly_loss_ce':
        loss_fn = PolyLoss_CE(epsilon = args.eps).to(args.device)
    elif args.loss_fn == 'poly_loss_fl':
        loss_fn = PolyLoss_FL(epsilon=args.eps,gamma = 2.0).to(args.device)
    else:
        raise ValueError('not support this loss : %s'%args.loss_fn)
    getEF(model,args,args.test_path,save_path,args.device,args.debug,args.batch_size,args.A2_limit,loss_fn,args.EF_rates,flag = '_' + args.test_name)
    # getEFMultiPose(model,args,args.test_path,save_path,args.debug,args.batch_size,loss_fn,rates = args.EF_rates,flag = '_10' + args.test_name,pose_num = 10)
if '__main__' == __name__:
    from torch import distributed as dist
    import torch.multiprocessing as mp
    from dist_utils import *
    def get_args_from_json(json_file_path, args_dict):
        import json
        summary_filename = json_file_path
        with open(summary_filename) as f:
            summary_dict = json.load(fp=f)
        for key in summary_dict.keys():
            args_dict[key] = summary_dict[key]
        return args_dict
    parser = argparse.ArgumentParser(description='json param')
    parser.add_argument('--local_rank', default=-1, type=int) 
    parser.add_argument("--json_path", help="file path of param", type=str, \
        default='/home/caoduanhua/score_function/GNN/GNN_graphformer_pyg/new_data_train_keys/config_files/gnn_edge_3d_pos_screen_dgl_FP_pose_enhanced_challenge_cross_10_pose_search.json')
    args = parser.parse_args()
    local_rank = args.local_rank
    # label_smoothing# temp_args = parser.parse_args()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(args_dict['json_path'], args_dict)
    args = argparse.Namespace(**args)
    # 下面这个参数需要加上，torch内部调用多进程时，会使用该参数，对每个gpu进程而言，其local_rank都是不同的；
    args.local_rank = local_rank
    if args.ngpu>0:
        cmd = get_available_gpu(num_gpu=args.ngpu, min_memory=30000, sample=3, nitro_restriction=False, verbose=True)
        if cmd[-1] == ',':
            os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES']=cmd
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    from torch.multiprocessing import Process
    world_size = args.ngpu
    processes = []
    for rank in range(world_size):
        p = Process(target=run, args=(rank, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()