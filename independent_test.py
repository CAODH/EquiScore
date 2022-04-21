# from gnn_meta_learning.meta_learning import meta_learning
import sys
# sys.path.append("/home/caoduanhua/score_function/GNN/")
# sys.path.append("/home/caoduanhua/score_function/GNN/GNN_graphformer_pyg")
import pickle
from gnn import gnn
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
# import sys
# sys.path.append("/home/duanhua/projects/GNN_graphformer")
# print(os.path.abspath(os.path.dirname(__file__)))
# print(os.getcwd())
def retract():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
    parser.add_argument("--epoch", help="epoch", type=int, default = 300)
    parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
    parser.add_argument("--batch_size", help="batch_size", type=int, default = 12)
    parser.add_argument("--num_workers", help="number of workers", type=int, default = 4)
    parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default =2)
    # parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
    parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
    parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
    parser.add_argument("--data_path", help="file path of dude data", type=str, default='/home/caoduanhua/score_function/data/general_refineset')
    #/home/jiangjiaxin/../../../
    parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default ='../train_result/graphnorm/dude/')
    parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 2.5)#4.0
    parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 4.0)#1.0
    parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.2)
    #args.attention_dropout_rate
    parser.add_argument("--attention_dropout_rate", help="attention_dropout_rate", type=float, default = 0.2)
    parser.add_argument("--train_keys", help="train keys", type=str, default='/home/caoduanhua/score_function/GNN/GNN_graphformer_pyg/dude_keys/train_keys.pkl')
    parser.add_argument("--test_keys", help="test keys", type=str, default='/home/caoduanhua/score_function/data/independent/pocket_pcba_full')
    #add by caooduanhua
    # self.fundation_model = args.fundation_model
    parser.add_argument("--fundation_model", help="what kind of model to use : paper or graphformer", type=str, default='graphformer')
    parser.add_argument("--layer_type", help="what kind of layer to use :GAT_gata,MH_gate,transformer_gate,graphformer", type=str, default='GAT_gate')
    parser.add_argument("--loss_fn", help="what kind of loss_fn to use : bce_loss facal_loss mse_loss ", type=str, default='bce_loss')
    # args.gate
    parser.add_argument("--only_adj2", help="adj2 only have 0 1 ", action = 'store_true')
    parser.add_argument("--only_dis_adj2", help="sdj2 only have distance info ", action = 'store_true')
    parser.add_argument("--share_layer", help="select share layers with h1 h2 or not ", action = 'store_false')
    parser.add_argument("--use_adj", help="select sampler in train stage ", action = 'store_false')
    parser.add_argument("--mode", help="what kind of mode to training : only h1 to training or h1 h2 to training [1_H,2_H] ", type=str, default='1_H')
    parser.add_argument("--n_in_feature", help="dim before layers to tranform dim in paper model", type=int, default = 80)
    parser.add_argument("--n_out_feature", help="dim in layers", type=int, default = 80)
    parser.add_argument("--ffn_size", help="ffn dim in transformer type layers", type=int, default = 280)
    parser.add_argument("--head_size", help="multihead attention", type=int, default = 8)
    parser.add_argument("--patience", help="patience for early stop", type=int, default = 50)
    parser.add_argument("--gate", help="gate mode for Transformer_gate", action = 'store_true')
    parser.add_argument("--debug", help="debug mode for check", action = 'store_true')
    parser.add_argument("--test", help="independent tests or not ", action = 'store_true')
    parser.add_argument("--sampler", help="select sampler in train stage ", action = 'store_true')
    parser.add_argument("--A2_limit", help="select add a A2adj strong limit  in model", action = 'store_true')
    parser.add_argument("--test_path", help="test keys", type=str, default='/home/caoduanhua/score_function/data/independent/pocket_pcba_full/pocket')
    #/home/caoduanhua/score_function/data/independent/dekois_pocket
    #/home/caoduanhua/score_function/data/independent/dude_pocket

    parser.add_argument("--path_data_dir", help="saved shortest path data", type=str, default='../../data/pocket_data_path')


    parser.add_argument("--EF_rates", help="eval EF value in different percentage",nargs='+', type=float, default = [0.001,0.002,0.005,0.01,0.02,0.05])
    #parser.add_argument('--nargs-int-type', nargs='+', type=int)
    parser.add_argument("--multi_hop_max_dist", help="how many edges to use in multi-hop edge bias", type=int, default = 10)
    parser.add_argument("--edge_type", help="use multi-hop edge or not:single or multi_hop ", type=str, default='single')
    parser.add_argument("--rel_pos_bias", help="add rel_pos_bias or not default not ", action = 'store_true') 
    parser.add_argument("--edge_bias", help="add edge_bias or not default not ", action = 'store_true')       
    parser.add_argument("--rel_3d_pos_bias", help="add rel_3d_pos_bias or not default not ", action = 'store_true')        
    parser.add_argument("--in_degree_bias", help="add in_degree_bias or not default not ", action = 'store_true')  
    parser.add_argument("--out_degree_bias", help="add out_degree_bias or not default not ", action = 'store_true')          
    # save_model
    parser.add_argument("--hot_start", help="hot start", action = 'store_true')
    parser.add_argument("--save_model", help="hot start", type=str, default='/home/caoduanhua/score_function/GNN/train_result/graphnorm/graphformer/GAT_gate/2021-12-24-05-33-55/save_best_model.pt')
    parser.add_argument("--lr_decay", help="use lr decay ", action = 'store_true')  
    # auxiliary_loss
    parser.add_argument("--auxiliary_loss", help="use lr decay ", action = 'store_true') 
    parser.add_argument("--r_drop", help="use lr decay ", action = 'store_true') 
    parser.add_argument("--deta_const", help="const deta ", action = 'store_true') 
    parser.add_argument("--alpha", help="use lr decay ", type = int,default = 5) 
    parser.add_argument("--norm_type",help = 'select norm type in gnnyou can select  ln or gn ',type = str,choices=['gn_mul','gn_iter','ln'],default = 'gn_mul')
    parser.add_argument("--att_mode", help="attention mode ", default= 'SA',type=str,choices=['DSA','SA']) 
    #pred_mode
    parser.add_argument("--pred_mode",help = 'select nodes to be used  for prediction of graph ',type = str,choices= ['ligand','protein','supernode'],default = 'ligand')
    #set super node
    parser.add_argument("--supernode", help="const deta ", action = 'store_true')
    parser.add_argument("--embed_graph_mode",help = 'select nodes to be used  for prediction of graph ',type = str,choices= ['only_ligand','ligand_protein','dynamic_adj'],default = 'ligand_protein')
    parser.add_argument("--grad_sum", help="grad sum  ", action = 'store_true')
    parser.add_argument("--virtual_aromatic_atom", help="virtual_aromatic_atom center  ", action = 'store_true')
    # N_atom_features = 28
    parser.add_argument("--FP", help="use attentive FP feat", action = 'store_true')
    parser.add_argument("--dis_adj2_with_adj1", help="like name", action = 'store_true')
    parser.add_argument("--seed", help="use lr decay ", type = int,default = 42) 
    # single_graph
    parser.add_argument("--save_logits", help="save_logits", action = 'store_true')
    parser.add_argument("--embed_graph", help="only use one graph to learning  ", default = 'double_graph' ,choices = ['single_graph','double_graph','embed_graph_once'])
    #这套参数默认是paper+ GAT——gate without attn_bias
    args = parser.parse_args()

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
    default='/home/caoduanhua/score_function/GNN/GNN_graphformer_pyg/train_keys/config_files/train.json')
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
best_name = '/home/caoduanhua/score_function/GNN/train_result/ligand_shortest_path_bias/graphformer/GAT_gate/2022-04-11-09-29-29/save_best_model.pt'
save_path = best_name.replace('/save_best_model.pt','')
if not os.path.exists(save_path):
    os.makedirs(save_path)
#/home/caoduanhua/score_function/GNN/train_result/MSE/paper/GAT_gate/2021-12-08-10-28-48/save_best_model.pt
#/home/caoduanhua/score_function/GNN/train_result/MSE/paper/GAT_gate/2021-12-03-06-05-48/save_best_model.pt
args.best_name = best_name
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device
# device = torch.device('cpu')
model = utils.initialize_model(model, device, load_save_file = best_name )[0]
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
getEF(model,args,args.test_path,save_path,device,args.debug,args.batch_size,args.A2_limit,loss_fn,args.EF_rates,flag = '_pcba')
# getEF_from_MSE(model,args,args.test_path,save_path,device,args.debug,args.batch_size,args.A2_limit,loss_fn,args.EF_rates)
