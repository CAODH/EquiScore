import pickle
# import optuna
# from optuna.trial import TrialState
from gnn import gnn
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
# print(os.path.abspath(os.path.dirname(__file__)))
# print(os.getcwd())
def retract():
    pass
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
    # parser.add_argument("--epoch", help="epoch", type=int, default = 400)
    # parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
    # parser.add_argument("--batch_size", help="batch_size", type=int, default = 12)
    # parser.add_argument("--num_workers", help="number of workers", type=int, default = 8)
    # parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 2)
    # # parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
    # parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
    # parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
    # parser.add_argument("--data_path", help="file path of dude data", type=str, default='/home/caoduanhua/score_function/data/general_refineset')
    # #/home/jiangjiaxin/../../../
    # parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default ='/home/caoduanhua/score_function/GNN/train_result/same_nums_min_rmsd/')
    # parser.add_argument("--initial_mu", help="initial value of mu", type=float, default = 2.5)#4.0
    # parser.add_argument("--initial_dev", help="initial value of dev", type=float, default = 4.0)#1.0
    # parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default = 0.2)
    # #args.attention_dropout_rate
    # parser.add_argument("--attention_dropout_rate", help="attention_dropout_rate", type=float, default = 0.2)
    # parser.add_argument("--train_keys", help="train keys", type=str, default='/home/caoduanhua/score_function/GNN/GNN_graphformer_pyg/same_nums_min_rmsd_keys/train_keys.pkl')
    # parser.add_argument("--test_keys", help="test keys", type=str, default='/home/caoduanhua/score_function/GNN/GNN_graphformer_pyg/same_nums_min_rmsd_keys/test_keys.pkl')
    # #add by caooduanhua
    # # self.fundation_model = args.fundation_model
    # parser.add_argument("--fundation_model", help="what kind of model to use : paper or graphformer", type=str, default='graphformer')
    # parser.add_argument("--layer_type", help="what kind of layer to use :GAT_gata,MH_gate,transformer_gate,graphformer", type=str, default='GAT_gate')
    # parser.add_argument("--loss_fn", help="what kind of loss_fn to use : bce_loss facal_loss mse_loss ", type=str, default='bce_loss')
    # # args.gate
    # parser.add_argument("--only_adj2", help="adj2 only have 0 1 ", action = 'store_true')
    # parser.add_argument("--only_dis_adj2", help="sdj2 only have distance info ", action = 'store_true')
    # parser.add_argument("--share_layer", help="select share layers with h1 h2 or not ", action = 'store_false')
    # parser.add_argument("--use_adj", help="select sampler in train stage ", action = 'store_false')
    # parser.add_argument("--mode", help="what kind of mode to training : only h1 to training or h1 h2 to training [1_H,2_H] ", type=str, default='1_H')
    # parser.add_argument("--n_in_feature", help="dim before layers to tranform dim in paper model", type=int, default = 80)
    # parser.add_argument("--n_out_feature", help="dim in layers", type=int, default = 80)
    # parser.add_argument("--ffn_size", help="ffn dim in transformer type layers", type=int, default = 280)
    # parser.add_argument("--head_size", help="multihead attention", type=int, default = 8)
    # parser.add_argument("--patience", help="patience for early stop", type=int, default = 50)
    # parser.add_argument("--gate", help="gate mode for Transformer_gate", action = 'store_true')
    # parser.add_argument("--debug", help="debug mode for check", action = 'store_true')
    # parser.add_argument("--test", help="independent tests or not ", action = 'store_false')
    # parser.add_argument("--sampler", help="select sampler in train stage ", action = 'store_true')
    # parser.add_argument("--A2_limit", help="select add a A2adj strong limit  in model", action = 'store_true')
    # parser.add_argument("--test_path", help="test keys", type=str, default='/home/caoduanhua/score_function/data/independent/dude_pocket')
    # parser.add_argument("--path_data_dir", help="saved shortest path data", type=str, default='../../data/pocket_data_path')


    # parser.add_argument("--EF_rates", help="eval EF value in different percentage",nargs='+', type=float, default = 0.01)
    # #parser.add_argument('--nargs-int-type', nargs='+', type=int)
    # parser.add_argument("--multi_hop_max_dist", help="how many edges to use in multi-hop edge bias", type=int, default = 10)
    # parser.add_argument("--edge_type", help="use multi-hop edge or not:single or multi_hop ", type=str, default='single')
    # parser.add_argument("--rel_pos_bias", help="add rel_pos_bias or not default not ", action = 'store_true') 
    # parser.add_argument("--edge_bias", help="add edge_bias or not default not ", action = 'store_true')       
    # parser.add_argument("--rel_3d_pos_bias", help="add rel_3d_pos_bias or not default not ", action = 'store_true')        
    # parser.add_argument("--in_degree_bias", help="add in_degree_bias or not default not ", action = 'store_true')  
    # parser.add_argument("--out_degree_bias", help="add out_degree_bias or not default not ", action = 'store_true')          
    # # save_model
    # parser.add_argument("--hot_start", help="hot start", action = 'store_true')
    # parser.add_argument("--save_model", help="hot start", type=str, default='/home/caoduanhua/score_function/GNN/train_result/pdb_screen/graphformer/GAT_gate/2022-03-07-05-48-46/save_best_model.pt')
    # parser.add_argument("--lr_decay", help="use lr decay ", action = 'store_true')  
    # # auxiliary_loss
    # parser.add_argument("--auxiliary_loss", help="use lr decay ", action = 'store_true') 
    # parser.add_argument("--r_drop", help="use lr decay ", action = 'store_true') 
    # parser.add_argument("--deta_const", help="const deta ", action = 'store_true') 
    # parser.add_argument("--alpha", help="use lr decay ", type = int,default = 5) 
    # parser.add_argument("--norm_type",help = 'select norm type in gnnyou can select  ln or gn ',type = str,choices=['gn_mul','gn_iter','ln'],default = 'gn_mul')
    # parser.add_argument("--att_mode", help="attention mode ", default= 'SA',type=str,choices=['DSA','SA']) 
    # #pred_mode
    # parser.add_argument("--pred_mode",help = 'select nodes to be used  for prediction of graph ',type = str,choices= ['ligand','protein','supernode'],default = 'ligand')
    # #set super node
    # parser.add_argument("--supernode", help="const deta ", action = 'store_true')
    # parser.add_argument("--embed_graph_mode",help = 'select nodes to be used  for prediction of graph ',type = str,choices= ['only_ligand','ligand_protein','dynamic_adj'],default = 'ligand_protein')
    # parser.add_argument("--grad_sum", help="grad sum  ", action = 'store_true')
    # parser.add_argument("--virtual_aromatic_atom", help="virtual_aromatic_atom center  ", action = 'store_true')
    # # N_atom_features = 28
    # parser.add_argument("--FP", help="use attentive FP feat", action = 'store_true')
    # parser.add_argument("--dis_adj2_with_adj1", help="like name", action = 'store_true')
    # parser.add_argument("--seed", help="use lr decay ", type = int,default = 42) 
    # # single_graph
    # parser.add_argument("--embed_graph", help="only use one graph to learning  ", default = 'double_graph' ,choices = ['single_graph','double_graph','embed_graph_once'])
    # #这套参数默认是paper+ GAT——gate without attn_bias
# import argparse

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
    default='/home/caoduanhua/score_function/GNN/GNN_graphformer_pyg/train_keys/config_files/train_add_reg_logk.json')

# label_smoothing# temp_args = parser.parse_args()
args_dict = vars(parser.parse_args())
args = get_args_from_json(args_dict['json_path'], args_dict)
args = argparse.Namespace(**args)
print (args)
def run(args):
    # seed_everything()
    seed_torch(seed = args.seed)
    args_dict = vars(args)
    if args.FP:
        args.N_atom_features = 39
    else:
        args.N_atom_features = 28
    #hyper parameters
    num_epochs = args.epoch
    lr = args.lr
    ngpu = args.ngpu
    batch_size = args.batch_size
    data_path = args.data_path
    save_dir = args.save_dir
    train_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    #make save dir if it doesn't exist
    if args.hot_start:
        if os.path.exists(args.save_model):

            save_path = args.save_model.replace('save_best_model.pt','')
        else:
            raise ValueError('save_model is not a valid file check it again!')
    else:

        save_path = save_dir+ args.fundation_model + '/'+ args.layer_type +'/'+ train_time
    # save_path = save_dir+ args.fundation_model + '/'+ args.layer_type +'/'+ train_time
    # print('save_path:',save_path)
    if not os.path.exists(save_path):
        os.system('mkdir -p ' + save_path)
    log_path = save_path+'/logs' 
    # model_path = save_dir+train_time+'/models' 
    #read data. data is stored in format of dictionary. Each key has information about protein-ligand complex.
    with open (args.train_keys, 'rb') as fp:
        train_keys = pickle.load(fp)
    train_keys,val_keys = random_split(train_keys, split_ratio=0.9, seed=0, shuffle=True)
    # print(sum([1 for key in val_keys if '_active' in key ])/len(val_keys),len(val_keys))
    with open (args.test_keys, 'rb') as fp:
        test_keys = pickle.load(fp)
    # test_keys = os.listdir(args.test_path)
    # train_keys = [args.data_path + i for i in train_keys]
    # test_keys = [args.data_path + i for i in test_keys]
    #print simple statistics about dude data and pdbbind data
    print (f'Number of train data: {len(train_keys)}')
    print (f'Number of val data: {len(val_keys)}')
    print (f'Number of test data: {len(test_keys)}')

    #initialize model
    if args.ngpu>0:
        cmd = get_available_gpu(num_gpu=args.ngpu, min_memory=10000, sample=3, nitro_restriction=False, verbose=True)
        # cmd = '1,'
        #cmd = utils.set_cuda_visible_device(args.ngpu)
        if cmd[-1] == ',':

            os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES']=cmd
        # print(cmd)


    model = gnn(args)

    print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    if args.hot_start:

        model ,opt_dict,epoch_start= utils.initialize_model(model, args.device,args.save_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # print('opt_dict: ',opt_dict)
        optimizer.load_state_dict(opt_dict)
        # print('optimizer: ',optimizer)
    else:

        model = utils.initialize_model(model, args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        epoch_start = 0
        write_log_head(args,log_path,model,train_keys,val_keys)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=10, epochs=10)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,200,250], gamma=0.5)
    

    #train val and test dataset

    train_dataset = graphformerDataset(train_keys,args, args.data_path,args.debug)#keys,args, data_dir,debug
    val_dataset = graphformerDataset(val_keys,args, args.data_path,args.debug)
    # test_dataset = graphformerDataset(test_keys,args, args.data_path,args.debug) 测试集看不出什么东西，直接忽略
    # test_dataset = MolDataset(test_keys, args.data_path,args.debug)
    if args.sampler:

        num_train_chembl = len([0 for k in train_keys if '_active' in k])
        num_train_decoy = len([0 for k in train_keys if '_active' not in k])
        train_weights = [1/num_train_chembl if '_active' in k else 1/num_train_decoy for k in train_keys]
        train_sampler = DTISampler(train_weights, len(train_weights), replacement=True)                     
        train_dataloader = DataLoaderX(train_dataset, args.batch_size, \
            shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn,\
            sampler = train_sampler,pin_memory=True,drop_last = True)#动态采样
    else:
        train_dataloader = DataLoaderX(train_dataset, args.batch_size, \
            shuffle=True, num_workers = args.num_workers, collate_fn=collate_fn,pin_memory=True)
    val_dataloader = DataLoaderX(val_dataset, args.batch_size, \
        shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn,pin_memory=True)
    # test_dataloader = DataLoaderX(test_dataset, args.batch_size, \
    #     shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn,pin_memory=True)  测试集看不出什么东西，直接忽略

    #optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #loss function
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
    best_loss = 1000000000#by caodunahua
    counter = 0
    
    for epoch in range(epoch_start,num_epochs):
        st = time.time()
        #collect losses of each iteration
        

        model,train_pred,train_losses,optimizer = train(model,args,optimizer,loss_fn,train_dataloader,auxiliary_loss)
        if args.lr_decay:
            scheduler.step()
        val_losses,val_true,val_pred = evaluator(model,val_dataloader,loss_fn,args)
        # test_losses,test_true,test_pred = evaluator(model,test_dataloader,loss_fn,args)
        train_losses = np.mean(np.array(train_losses))
        # test_losses = np.mean(np.array(test_losses))
        val_losses = np.mean(np.array(val_losses))
        #by caoduanhua under this line
        # train_auroc,train_adjust_logauroc,train_auprc,train_balanced_acc,train_acc,train_precision,train_sensitity,train_specifity,train_f1 = get_metrics(train_true,train_pred)
    #计算test
        if args.loss_fn == 'mse_loss':
            end = time.time()
            with open(log_path,'a') as f:
                f.write(str(epoch)+ '\t'+str(train_losses)+ '\t'+str(val_losses)+ '\t'+str(test_losses) + str(end-st)+ '\n')
                f.close()
        else:

            test_auroc,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(val_true,val_pred)

            end = time.time()
            with open(log_path,'a') as f:

                f.write(str(epoch)+ '\t'+str(train_losses)+ '\t'+str(val_losses)+ '\t'+str(0.0)\
                    #'\t'+str(train_auroc)+ '\t'+str(train_adjust_logauroc)+ '\t'+str(train_auprc)+ '\t'+str(train_balanced_acc)+ '\t'+str(train_acc)+ '\t'+str(train_precision)+ '\t'+str(train_sensitity)+ '\t'+str(train_specifity)+ '\t'+str(train_f1)\

                + '\t'+str(test_auroc)+ '\t'+str(test_adjust_logauroc)+ '\t'+str(test_auprc)+ '\t'+str(test_balanced_acc)+ '\t'+str(test_acc)+ '\t'+str(test_precision)+ '\t'+str(test_sensitity)+ '\t'+str(test_specifity)+ '\t'+str(test_f1) +'\t'\

                + str(end-st)+ '\n')
                f.close()
    #by caoduanhua 通过val 来获取最好的模型，而不是test
       #by caoduanhua 通过val 来获取最好的模型，而不是test
        counter +=1 
        if val_losses < best_loss:
            best_loss = val_losses
            counter = 0
            save_model(model,optimizer,args,epoch,save_path,mode = 'best')
        if counter > args.patience:
            save_model(model,optimizer,args,epoch,save_path,mode = 'early_stop')
            print('model early stop !')
            break
        if epoch == num_epochs-1:
            save_model(model,optimizer,args,epoch,save_path,mode = 'end')
    print('training done!')
    if args.test:
        model = gnn(args)
        device = args.device#("cuda:0" if torch.cuda.is_available() else "cpu")
        model = utils.initialize_model(model, device, load_save_file = save_path + '/save_best_model.pt')[0]
        EF_file = save_path +'/EF_test'
        getEF(model,args,args.test_path,save_path,device,args.debug,args.batch_size,args.A2_limit,loss_fn,args.EF_rates)
    
if '__main__' == __name__:

    run(args)
