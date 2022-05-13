import pickle
# import optuna
# from optuna.trial import TrialState
from gnn import gnn
from gnn_edge import gnn_edge
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
from dgl.dataloading import GraphDataLoader
from prefetch_generator import BackgroundGenerator
from graph_transformer_net import GraphTransformerNet
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())                            
from graphformer_dataset import graphformerDataset,  DTISampler
now = time.localtime()
from rdkit import RDLogger
# import torch.multiprocessing as mp
# mp.set_spawn_method("spawn")
RDLogger.DisableLog('rdApp.*')
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print (s)
os.chdir(os.path.abspath(os.path.dirname(__file__)))

parser = argparse.ArgumentParser(description='json param')
parser.add_argument("--json_path", help="file path of param", type=str, \
    default='/home/caoduanhua/score_function/GNN/GNN_graphformer_pyg/train_keys/config_files/gnn_edge_3d_pos_dgl.json')

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


    model = GraphTransformerNet(args) if args.gnn_model == 'graph_transformer_dgl' else None

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
        train_dataloader = GraphDataLoader(train_dataset, args.batch_size, \
            shuffle=False, num_workers = args.num_workers, collate_fn=train_dataset.collate,\
            sampler = train_sampler,pin_memory=True,drop_last = True)#动态采样
    else:
        train_dataloader = GraphDataLoader(train_dataset, args.batch_size, \
            shuffle=True, num_workers = args.num_workers, collate_fn=train_dataset.collate,pin_memory=True)
    val_dataloader = GraphDataLoader(val_dataset, args.batch_size, \
        shuffle=False, num_workers = args.num_workers, collate_fn=val_dataset.collate,pin_memory=True)
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
