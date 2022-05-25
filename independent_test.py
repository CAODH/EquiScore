# from gnn_meta_learning.meta_learning import meta_learning
import sys
# sys.path.append("/home/caoduanhua/score_function/GNN/")
# sys.path.append("/home/caoduanhua/score_function/GNN/GNN_graphformer_pyg")
import pickle
# from gnn import gnn
import time
import numpy as np
import utils
from utils import *
import torch.nn as nn
import torch
import warnings
import time
import os
from collections import defaultdict
import argparse
import time
from torch.utils.data import DataLoader                                     
from graphformer_dataset import graphformerDataset, collate_fn, DTISampler
# from getdataloader import get_img_dataloader,random_pairs_of_minibatches_by_domainperm,get_naive_dataloader
# from meta_learning import meta_learning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
now = time.localtime()
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print (s)
os.chdir(os.path.abspath(os.path.dirname(__file__)))


def get_args_from_json(json_file_path, args_dict):
    import json
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)
    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]
    return args_dict
parser = argparse.ArgumentParser(description='json param')
parser.add_argument("--json_path", help="file path of param", type=str, \
    default='/home/caoduanhua/score_function/GNN/GNN_graphformer_pyg/train_keys/config_files/ligand_shortest_path_3d_pos.json')
# temp_args = parser.parse_args()
args_dict = vars(parser.parse_args())
args = get_args_from_json(args_dict['json_path'], args_dict)
args = argparse.Namespace(**args)
seed_torch(args.seed)
args_dict = vars(args)
if args.FP:
    args.N_atom_features = 39
else:
    args.N_atom_features = 28
model = gnn(args) 
# print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
# save_path = args.save_dir+ 'independent_test' +'/' + time.strftime('%Y-%m-%d-%H-%M-%S')
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
if args.ngpu>0:
    cmd = get_available_gpu(num_gpu=1, min_memory=6000, sample=3, nitro_restriction=False, verbose=True)
    # cmd = '1,'
    os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
    if cmd[-1] == ',':
            os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
    else:
        os.environ['CUDA_VISIBLE_DEVICES']=cmd
    print(cmd)
best_name = '/home/caoduanhua/score_function/GNN/train_result/ligand_shortest_path_3d_pos/graphformer/GAT_gate/2022-04-11-11-16-27/save_best_model.pt'
save_path = best_name.replace('/save_best_model.pt','')
if not os.path.exists(save_path):
    os.makedirs(save_path)
#/home/caoduanhua/score_function/GNN/train_result/MSE/paper/GAT_gate/2021-12-08-10-28-48/save_best_model.pt
#/home/caoduanhua/score_function/GNN/train_result/MSE/paper/GAT_gate/2021-12-03-06-05-48/save_best_model.pt
args.best_name = best_name
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device
# device = torch.device('cpu')
model = utils.initialize_model(model, device, args,load_save_file = best_name )[0]
if args.loss_fn == 'bce_loss':
    loss_fn = nn.BCELoss().to(args.device)# 
elif args.loss_fn == 'focal_loss':
    loss_fn = FocalLoss().to(args.device)
elif args.loss_fn == 'cross_entry':
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smothing).to(args.device)
elif args.loss_fn == 'mse_loss':
    loss_fn = nn.MSELoss().to(args.device)
else:
    raise ValueError('not support this loss : %s'%args.loss_fn)
# EF_file = save_path +'/EF_test' + time.strftime('%Y-%m-%d-%H-%M-%S')
getEF(model,args,args.test_path,save_path,device,args.debug,args.batch_size,args.A2_limit,loss_fn,args.EF_rates,flag = '_dekois')
# getEF_from_MSE(model,args,args.test_path,save_path,device,args.debug,args.batch_size,args.A2_limit,loss_fn,args.EF_rates)
