import pickle
from gnn import gnn
import time
import numpy as np
import utils
from utils import *
import torch.nn as nn
import torch
import time
import os
from train import run
from collections import defaultdict
import argparse
import time
from torch.utils.data import DataLoader                                     
# from dataset import MolDataset, collate_fn, DTISampler
from graphformer_dataset import graphformerDataset, collate_fn, DTISampler
now = time.localtime()
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print (s)
parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default = 10000)
parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 64)
parser.add_argument("--num_workers", help="number of workers", type=int, default = 8)
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 2)
# parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
parser.add_argument("--data_path", help="file path of dude data", type=str, default='/home/chengeng/project/scorefunction/pdbbind_cross_decoy/data/pocket_data')
#/home/jiangjiaxin/../../../
parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default ='../train_result/GNN_crose_decoy/')
parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 4.0)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 1.0)
parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.1)
#args.attention_dropout_rate
parser.add_argument("--attention_dropout_rate", help="attention_dropout_rate", type=float, default = 0.1)
parser.add_argument("--train_keys", help="train keys", type=str, default='./keys/train_keys.pkl')
parser.add_argument("--test_keys", help="test keys", type=str, default='./keys/test_keys.pkl')
#add by caooduanhua
# self.fundation_model = args.fundation_model
parser.add_argument("--fundation_model", help="what kind of model to use : paper or graphformer", type=str, default='paper')
parser.add_argument("--layer_type", help="what kind of layer to use :GAT_gata,MH_gate,transformer_gate,graphformer", type=str, default='GAT_gate')
parser.add_argument("--loss_fn", help="what kind of loss_fn to use : bce_loss facal_loss ", type=str, default='bce_loss')
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
parser.add_argument("--patience", help="patience for early stop", type=int, default = 5)
parser.add_argument("--gate", help="gate mode for Transformer_gate", action = 'store_true')
parser.add_argument("--debug", help="debug mode for check", action = 'store_true')
parser.add_argument("--test", help="independent tests or not ", action = 'store_true')
parser.add_argument("--sampler", help="select sampler in train stage ", action = 'store_true')
parser.add_argument("--A2_limit", help="select add a A2adj strong limit  in model", action = 'store_true')
parser.add_argument("--test_path", help="test keys", type=str, default='/home/duanhua/data/pocket_sample_70w/train')
parser.add_argument("--path_data_dir", help="saved shortest path data", type=str, default='../../data/pocket_data_path')
parser.add_argument("--EF_rates", help="eval EF value in different percentage",nargs='+', type=float, default = 0.01)
#parser.add_argument('--nargs-int-type', nargs='+', type=int)
parser.add_argument("--multi_hop_max_dist", help="how many edges to use in multi-hop edge bias", type=int, default = 10)
parser.add_argument("--edge_type", help="use multi-hop edge or not:single or multi_hop ", type=str, default='single')
parser.add_argument("--rel_pos_bias", help="add rel_pos_bias or not default not ", action = 'store_true') 
parser.add_argument("--edge_bias", help="add edge_bias or not default not ", action = 'store_true')       
parser.add_argument("--rel_3d_pos_bias", help="add rel_3d_pos_bias or not default not ", action = 'store_true')        
parser.add_argument("--in_degree_bias", help="add in_degree_bias or not default not ", action = 'store_true')  
parser.add_argument("--out_degree_bias", help="add out_degree_bias or not default not ", action = 'store_true')          
    
#这套参数默认是paper+ GAT——gate without attn_bias 
args = parser.parse_args()
#what do you want search 


if '__main__' == __name__:
    for lr in [0.0005,0.001]:
        for n_graph_layer in [1,2]:
            for d_graph_layer in [35,70]:
                for n_FC_layer in [1,2]:
                    for d_FC_layer in [16,32,64]:
                            args.lr = lr
                            args.n_graph_layer = n_graph_layer
                            args.d_graph_layer = d_graph_layer
                            args.n_FC_layer = n_FC_layer
                            args.d_FC_layer = d_FC_layer
                            # args.dropout_rate = dropout_rate
                            run(args)
