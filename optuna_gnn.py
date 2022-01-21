import pickle
import optuna
from optuna.trial import TrialState
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
print (s)
os.chdir(os.path.abspath(os.path.dirname(__file__)))
# print(os.path.abspath(os.path.dirname(__file__)))
# print(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="learning rate", type=float, default = 0.0001)
parser.add_argument("--epoch", help="epoch", type=int, default = 1000)
parser.add_argument("--ngpu", help="number of gpu", type=int, default = 1)
parser.add_argument("--batch_size", help="batch_size", type=int, default = 128)
parser.add_argument("--num_workers", help="number of workers", type=int, default = )
parser.add_argument("--n_graph_layer", help="number of GNN layer", type=int, default = 2)
# parser.add_argument("--d_graph_layer", help="dimension of GNN layer", type=int, default = 140)
parser.add_argument("--n_FC_layer", help="number of FC layer", type=int, default = 4)
parser.add_argument("--d_FC_layer", help="dimension of FC layer", type=int, default = 128)
parser.add_argument("--data_path", help="file path of dude data", type=str, default='/home/caoduanhua/score_function/data/pocket_data')
#/home/jiangjiaxin/../../../
parser.add_argument("--save_dir", help="save directory of model parameter", type=str, default ='../train_result/optuna/')
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
parser.add_argument("--patience", help="patience for early stop", type=int, default = 50)
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
# save_model
parser.add_argument("--save_model", help="hot start", type=str, default='')
#这套参数默认是paper+ GAT——gate without attn_bias 
args = parser.parse_args()
print (args)
def objective(trial):
    # args_dict = vars(args)
    #hyper parameters
    num_epochs = args.epoch
    lr = args.lr
    ngpu = args.ngpu
    batch_size = args.batch_size
    data_path = args.data_path
    save_dir = args.save_dir
    train_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    #make save dir if it doesn't exist
    save_path = save_dir+ args.fundation_model + '/'+ args.layer_type +'/'+ train_time
    print('save_path:',save_path)
    if not os.path.exists(save_path):
        os.system('mkdir -p ' + save_path)
    if os.path.exists(args.save_model):
        log_path = args.save_model.replace('save_best_model.pt','logs')
    else:

        log_path = save_path+'/logs' 
    # model_path = save_dir+train_time+'/models' 
    #read data. data is stored in format of dictionary. Each key has information about protein-ligand complex.
    with open (args.train_keys, 'rb') as fp:
        train_keys = pickle.load(fp)
    train_keys,val_keys = random_split(train_keys, split_ratio=0.9, seed=0, shuffle=True)
    with open (args.test_keys, 'rb') as fp:
        test_keys = pickle.load(fp)
    # test_keys = os.listdir(args.test_path)
    # train_keys = [args.data_path + i for i in train_keys]
    # test_keys = [args.data_path + i for i in test_keys]
    #print simple statistics about dude data and pdbbind data
    print (f'Number of train data: {len(train_keys)}')
    print (f'Number of test data: {len(val_keys)}')
    print (f'Number of test data: {len(test_keys)}')

    #initialize model
    if args.ngpu>0:
        cmd = get_available_gpu(num_gpu=args.ngpu, min_memory=1000, sample=3, nitro_restriction=False, verbose=True)
        # cmd = '1,'
        #cmd = utils.set_cuda_visible_device(args.ngpu)
        if cmd[-1] == ',':

            os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES']=cmd
        print(cmd)
    #---------------------------------------------------------
    #configure the optuna !
    n_graph_layers = trial.suggest_int("n_layers", 1,3,step = 1)
    n_FC_layer = trial.suggest_int("n_FC_layer", 1,3,step = 1)

    d_FC_layer = trial.suggest_categorical("d_FC_layer", [32,64,128,256])

    dropout_rate = trial.suggest_categorical("dropout_rate", [0.1,0.2,0.5])
    # head_size = trial.suggest_int("head_size", 1,2,3)
    # n_in_feature = trial.suggest_int("n_in_feature", 1,2,3)
    # n_out_feature = trial.suggest_int("n_out_feature", 1,2,3)
    args.n_graph_layers = n_graph_layers
    args.n_FC_layer = n_FC_layer
    args.d_FC_layer = d_FC_layer
    args.dropout_rate = dropout_rate

#---------------------------------------------------------------
    args_dict = vars(args)
    model = gnn(args)

    print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    if os.path.exists(args.save_model):

        model ,opt_dict,epoch_start= utils.initialize_model(model, args.device,args.save_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # print('opt_dict: ',opt_dict)
        optimizer.load_state_dict(opt_dict)
        # print('optimizer: ',optimizer)
    else:

        model = utils.initialize_model(model, args.device)
        # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD","AdamW"])
        # lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        epoch_start = 0
    with open(log_path,'a')as f:
        f.write(f'Number of train data: {len(train_keys)}' +'\n'+ f'Number of test data: {len(val_keys)}' + '\n')
        f.write(f'number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}' +'\n')

        for item in args_dict.keys():
            f.write(item + ' : '+str(args_dict[item]) + '\n')
        f.write('epoch'+'\t'+'train_loss'+'\t'+'val_loss'+'\t'+'test_loss' #'\t'+'train_auroc'+ '\t'+'train_adjust_logauroc'+ '\t'+'train_auprc'+ '\t'+'train_balanced_acc'+ '\t'+'train_acc'+ '\t'+'train_precision'+ '\t'+'train_sensitity'+ '\t'+'train_specifity'+ '\t'+'train_f1'+ '\t'\
        + '\t' + 'test_auroc'+ '\t'+'test_adjust_logauroc'+ '\t'+'test_auprc'+ '\t'+'test_balanced_acc'+ '\t'+'test_acc'+ '\t'+'test_precision'+ '\t'+'test_sensitity'+ '\t'+'test_specifity'+ '\t'+'test_f1' +'\t' +'time'+ '\n')
        f.close()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10)
    #train val and test dataset

    train_dataset = graphformerDataset(train_keys,args, args.data_path,args.debug)#keys,args, data_dir,debug
    val_dataset = graphformerDataset(val_keys,args, args.data_path,args.debug)
    test_dataset = graphformerDataset(test_keys,args, args.data_path,args.debug)
    # test_dataset = MolDataset(test_keys, args.data_path,args.debug)
    if args.sampler:

        num_train_chembl = len([0 for k in train_keys if '_active' in k])
        num_train_decoy = len([0 for k in train_keys if '_active' not in k])
        train_weights = [1/num_train_chembl if '_active' in k else 1/num_train_decoy for k in train_keys]
        train_sampler = DTISampler(train_weights, len(train_weights), replacement=True)                     
        train_dataloader = DataLoader(train_dataset, args.batch_size, \
            shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn,\
            sampler = train_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, args.batch_size, \
            shuffle=True, num_workers = args.num_workers, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, args.batch_size, \
        shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, args.batch_size, \
        shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn)

    #optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #loss function
    if args.loss_fn == 'bce_loss':

        loss_fn = nn.BCELoss()# 
    elif args.loss_fn == 'facal_loss':
        loss_fn = FocalLoss()
    else:
        raise ValueError('not support this loss : %s'%args.loss_fn)
    best_loss = 1000000000#by caodunahua
    counter = 0
    
    for epoch in range(epoch_start,num_epochs):
        st = time.time()
        #collect losses of each iteration
        train_losses = [] 
        train_true = []
        train_pred = []
        model.train()
        for i_batch, sample in enumerate(train_dataloader):
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
            # print('pred shape : ',pred.shape)
            # print('pred ',pred.shape)
            loss = loss_fn(pred, sample.Y.to(pred.device)) 
            loss.backward()
            optimizer.step()
            train_losses.append(loss.data.cpu().numpy())
            train_true.append(sample.Y.data.cpu().numpy())
            if pred.dim() ==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            train_pred.append(pred.data.cpu().numpy())
        val_losses,val_true,val_pred = evaluator(model,val_dataloader,loss_fn,args)
        test_losses,test_true,test_pred = evaluator(model,test_dataloader,loss_fn,args)
        train_losses = np.mean(np.array(train_losses))
        test_losses = np.mean(np.array(test_losses))
        val_losses = np.mean(np.array(val_losses))
        scheduler.step()
        #by caoduanhua under this line
        # train_auroc,train_adjust_logauroc,train_auprc,train_balanced_acc,train_acc,train_precision,train_sensitity,train_specifity,train_f1 = get_metrics(train_true,train_pred)
    #计算test
        test_auroc,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(test_true,test_pred)
        
        #早停优化
        trial.report(test_auprc, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        end = time.time()
        with open(log_path,'a') as f:

            f.write(str(epoch)+ '\t'+str(train_losses)+ '\t'+str(val_losses)+ '\t'+str(test_losses)\
                #'\t'+str(train_auroc)+ '\t'+str(train_adjust_logauroc)+ '\t'+str(train_auprc)+ '\t'+str(train_balanced_acc)+ '\t'+str(train_acc)+ '\t'+str(train_precision)+ '\t'+str(train_sensitity)+ '\t'+str(train_specifity)+ '\t'+str(train_f1)\

            + '\t'+str(test_auroc)+ '\t'+str(test_adjust_logauroc)+ '\t'+str(test_auprc)+ '\t'+str(test_balanced_acc)+ '\t'+str(test_acc)+ '\t'+str(test_precision)+ '\t'+str(test_sensitity)+ '\t'+str(test_specifity)+ '\t'+str(test_f1) +'\t'\

            + str(end-st)+ '\n')
            f.close()
    #by caoduanhua 通过val 来获取最好的模型，而不是test
        counter +=1 
        if val_losses < best_loss:
            best_loss = val_losses
            counter = 0
            best_name = save_path + '/save_best_model'+'.pt'
            if args.debug:
                best_name = save_path + '/save_best_model_debug'+'.pt'
    #         name = save_dir + '/save_best_model'+'.pt'
            torch.save({'model':model.module.state_dict() if isinstance(model,nn.DataParallel) else model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'epoch':epoch}, best_name)
        
        if counter > args.patience:
            early_name = save_path + '/save_early_stop_model'+str(epoch)+'.pt'
            if args.debug:
                early_name = save_path + '/save_early_stop_model_debug'+str(epoch)+'.pt'
    #         name = save_dir + '/save_early_stop_model'+'.pt'
            torch.save({'model':model.module.state_dict() if isinstance(model,nn.DataParallel) else model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'epoch':epoch}, early_name)
            print('model early stop !')
            return test_auprc
        if epoch == num_epochs-1:
            end_name = save_path + '/save_end_model'+str(epoch)+'.pt'
            if args.debug:
                end_name = save_path + '/save_end_model_debug'+str(epoch)+'.pt'
    #         name = save_dir + '/save_early_stop_model'+'.pt'
            torch.save({'model':model.module.state_dict() if isinstance(model,nn.DataParallel) else model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'epoch':epoch}, end_name)
    
    # print('training done!')

        # name = save_dir + '/save_'+str(epoch)+'.pt'
        # torch.save(model.state_dict(), name)
    if args.test:

        model = gnn(args)
        # print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = utils.initialize_model(model, device, load_save_file = best_name )
        EF_file = save_path +'/EF_test'
        getEF(model,args,args.test_path,EF_file,device,args.debug,args.batch_size,args.A2_limit,loss_fn,args.EF_rates)
        #
    return test_auprc
        
if '__main__' == __name__:

    # run()
    study = optuna.create_study(direction="maximize",storage='sqlite:///param_optuna.db',study_name='paper',load_if_exists = True)#有默认的剪枝策略？
    study.optimize(objective, n_trials=108)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))