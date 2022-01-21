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
from collections import defaultdict
import argparse
import time
from torch.utils.data import DataLoader                                     
from graphformer_dataset import graphformerDataset, collate_fn, DTISampler
now = time.localtime()
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
import warnings
warnings.filterwarnings('ignore')
now = time.localtime()
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print (s)
os.chdir(os.path.abspath(os.path.dirname(__file__)))
# import sys
# sys.path.append("/home/duanhua/projects/GNN_graphformer")
# print(os.path.abspath(os.path.dirname(__file__)))
# print(os.getcwd())
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
parser.add_argument("--data_path", help="file path of dude data", type=str, default='/home/duanhua/data/pocket_data')
#/home/jiangjiaxin/../../../
parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default ='../train_result/mata_learning/')
parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 4.0)
parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 1.0)
parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.1)
#args.attention_dropout_rate
parser.add_argument("--attention_dropout_rate", help="attention_dropout_rate", type=float, default = 0.1)
parser.add_argument("--train_keys", help="train keys", type=str, default='/home/duanhua/projects/GNN_graphformer/keys/train_keys.pkl')
parser.add_argument("--test_keys", help="test keys", type=str, default='/home/duanhua/projects/GNN_graphformer/keys/test_keys.pkl')
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
parser.add_argument("--test_path", help="test keys", type=str, default='/home/duanhua/data/pocket_data')
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
# select train mode
parser.add_argument("--train_mode", help="select to meta-learning or anather mode ", type=str, default='meta')
#set meta learning inner train step size
parser.add_argument("--step_per_epoch", help="inner steps in meta mode", type=int, default = 200)
#inner_lr
parser.add_argument("--inner_lr", help="inner lr", type=float, default = 0.0001)
# mldg_beta
parser.add_argument("--mldg_beta", help="inner loss weight to add to sum objective ", type=float, default = 1.0)
args = parser.parse_args()
def get_logits(model,args,test_path,device,debug,batch_size,loss_fn):
        ##先整理做一遍 Logit和auc
        with open(args.train_keys,'rb') as f:
            test_keys = pickle.load(f)



        pros = defaultdict(list)
        for key in test_keys:
            key_split = key.split('_')
            if 'active' in key_split:
                pros[key_split[0]].insert(0,key)
            else:#阳性标签排在前面
                pros[key_split[0]].append(key)
            #agg sme pro
        for pro in pros.keys():
            test_keys_pro = pros[pro]
            if test_keys_pro is None:
                continue
            test_dataset = graphformerDataset(test_keys_pro,args, test_path,debug)
            test_dataloader = DataLoader(test_dataset, batch_size = batch_size, \
            shuffle=False, num_workers = 8, collate_fn=collate_fn)
            model.eval()
            Y_sum = 0
            test_losses,test_true,test_pred = [],[],[]
            for i_batch, sample in enumerate(test_dataloader):
                model.zero_grad()
                
                #train neural network
                data_flag = []
                data = []
                for i in sample.get_att():
                    if type(i) is torch.Tensor:
                        data.append(i.to(device))
                        data_flag.append(1)
                    else:
                        data_flag.append(None)
                # num_flag = sum(data_flag)
                pred = model(args.A2_limit,data_flag,*data)

                # loss = loss_fn(pred, sample.Y.to(device)) 
                Y_sum += torch.sum(sample.Y)#阳性标签个数
                #collect loss, true label and predicted label
                # print(loss)
                # test_losses.append(loss.data.cpu().numpy())
                test_true.append(sample.Y.data.cpu().numpy())
                print(pred.shape)
                test_pred.append(pred.data.cpu().numpy())
                #if i_batch>10 : break
            actions = int(Y_sum)
            try:

                test_auroc,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(test_true,test_pred)
                test_pred = np.concatenate(np.array(test_pred), 0)
                with open('/home/duanhua/projects/train_result/GNN_DTI_fix/logits/dude_train_{}_logits'.format(pro),'wb') as f:
                    pickle.dump((test_pred,actions,test_auroc,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1),f)
                    f.close()
            except:
                test_pred = np.concatenate(np.array(test_pred), 0)
                with open('/home/duanhua/projects/train_result/GNN_DTI_fix/logits/dude_train_{}_logits'.format(pro),'wb') as f:
                    pickle.dump((test_pred,actions),f)
                    f.close()

print (args)
model = gnn(args)
# print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
save_path = args.save_dir + '/logits'
if not os.path.exists(save_path):
    os.makedirs(save_path)

best_name = '/home/duanhua/projects/train_result/mata_learning/graphformer/GAT_gate/2021-09-23-21-45-20/save_best_model.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.initialize_model(model, device, load_save_file = best_name )
loss_fn = nn.BCELoss()
EF_file = save_path +'/EF_test' + time.strftime('%Y-%m-%d-%H-%M-%S')

get_logits(model,args,args.test_path,device,args.debug,args.batch_size,loss_fn)
      