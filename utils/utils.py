import numpy as np
import torch
from torch import distributed as dist
import os.path
import time
import torch.nn as nn
from rdkit.ML.Scoring.Scoring import CalcBEDROC
from collections import defaultdict
from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve
from sklearn.metrics import accuracy_score,auc,balanced_accuracy_score
from sklearn.metrics import recall_score,precision_score,precision_recall_curve
from sklearn.metrics import confusion_matrix,f1_score
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())                            
from dataset.dataset import ESDataset,DTISampler
from utils.dist_utils import *
N_atom_features = 28
from scipy.spatial import distance_matrix
def get_args_from_json(json_file_path, args_dict):
    """"
    docstring:
        use this function to update the args_dict from a json file if you want to use a json file save parameters 
    input:
        json_file_path: string
            json file path
        args_dict: args dict
            dict

    output:
        args dict
    """

    import json
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)
    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]
    return args_dict

def initialize_model(model, device, args,load_save_file = False,init_classifer = True):
    """ initialize the model parameters or load the model from a saved file"""
    for param in model.parameters():
        if param.dim() == 1:
            continue
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
            

    if load_save_file:
        state_dict = torch.load(load_save_file,map_location = 'cpu')
        model_dict = state_dict['model']
        model_state_dict = model.state_dict()
        model_dict = {k:v for k,v in model_dict.items() if k in model_state_dict}
        model_state_dict.update(model_dict)
        model.load_state_dict(model_state_dict) 
        
        optimizer =state_dict['optimizer']
        epoch = state_dict['epoch']
        print('load save model!')
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
      
        model = model.cuda(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                     device_ids=[args.local_rank], 
                                                     output_device=args.local_rank, 
                                                     find_unused_parameters=True, 
                                                     broadcast_buffers=False)
        if load_save_file:
            return model ,optimizer,epoch
        return model
    model.to(args.local_rank)
    if load_save_file:
        return model ,optimizer,epoch
    return model

def get_logauc(fp, tp, min_fp=0.001, adjusted=False):
    """"
    docstring:
        use this function to calculate logauc 
    input:
        fp: list
            false positive
        tp: list
            true positive

    output: float
        logauc
    """
    
    lam_index = np.searchsorted(fp, min_fp)
    y = np.asarray(tp[lam_index:], dtype=np.double)
    x = np.asarray(fp[lam_index:], dtype=np.double)
    if (lam_index != 0):
        y = np.insert(y, 0, tp[lam_index - 1])
        x = np.insert(x, 0, min_fp)

    dy = (y[1:] - y[:-1])
    with np.errstate(divide='ignore'):
        intercept = y[1:] - x[1:] * (dy / (x[1:] - x[:-1]))
        intercept[np.isinf(intercept)] = 0.
    norm = np.log10(1. / float(min_fp))
    areas = ((dy / np.log(10.)) + intercept * np.log10(x[1:] / x[:-1])) / norm
    logauc = np.sum(areas)
    if adjusted:
        logauc -= 0.144620062  # random curve logAUC
    return logauc

def get_metrics(train_true,train_pred):
    # lr_decay
    """"
    docstring:
        calculate the metrics for the dataset
    input:
        train_true: list
            label
        train_pred: list
            predicted label

    output: list
        metrics
    """
    try:
        train_pred = np.concatenate(np.array(train_pred,dtype=object), 0).astype(np.float)
        train_true = np.concatenate(np.array(train_true,dtype=object), 0).astype(np.long)
    except:
        pass
    train_pred_label = np.where(train_pred > 0.5,1,0).astype(np.long)

    tn, fp, fn, tp = confusion_matrix(train_true,train_pred_label).ravel()
    train_auroc = roc_auc_score(train_true, train_pred) 
    train_acc = accuracy_score(train_true,train_pred_label)
    train_precision = precision_score(train_true,train_pred_label)
    train_sensitity = tp/(tp + fn)
    train_specifity = tn/(fp+tn)
    ps,rs,_ = precision_recall_curve(train_true,train_pred)
    train_auprc = auc(rs,ps)
    train_f1 = f1_score(train_true,train_pred_label)
    train_balanced_acc = balanced_accuracy_score(train_true,train_pred_label)
    fp,tp,_ = roc_curve(train_true,train_pred)
    train_adjusted_logauroc = get_logauc(fp,tp)
    # BEDROC
    sort_ind = np.argsort(train_pred)[::-1] # Descending order
    BEDROC = CalcBEDROC(train_true[sort_ind].reshape(-1, 1), 0, alpha = 80.5)
    return train_auroc,BEDROC,train_adjusted_logauroc,train_auprc,train_balanced_acc,train_acc,train_precision,train_sensitity,train_specifity,train_f1


def random_split(train_keys, split_ratio=0.9, seed=0, shuffle=True):
    """
    docstring:
        split the dataset into train and validation set by random sampling, this function not useful for new target protein prediction
    """
    
    
    dataset_size = len(train_keys)
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(split_ratio * dataset_size)
    train_idx, valid_idx = indices[:split], indices[split:]
    return [train_keys[i] for i in train_idx], [train_keys[i] for i in valid_idx]

def evaluator(model,loader,loss_fn,args,test_sampler):
    model.eval()
    with torch.no_grad():
        test_losses,test_true,test_pred = [], [],[]
        for i_batch, (g,full_g,Y) in enumerate(loader):
 
            model.zero_grad()
            g = g.to(args.local_rank,non_blocking=True)
            full_g = full_g.to(args.local_rank,non_blocking=True)
            Y = Y.long().to(args.local_rank,non_blocking=True)
            pred = model(g,full_g)
            loss = loss_fn(pred ,Y) 
 
            if args.ngpu >= 1:
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            # collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu >= 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)

            if pred.dim()==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu >= 1 else test_pred.append(pred.data)

        # gather ngpu result to single tensor
        if args.ngpu >= 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    return test_losses,test_true,test_pred
import copy
def train(model,args,optimizer,loss_fn,train_dataloader,scheduler):
    # collect losses of each iteration
    train_losses = [] 
    model.train()

    for i_batch, (g,full_g,Y) in enumerate(train_dataloader):
        g = g.to(args.device,non_blocking=True)
        full_g = full_g.to(args.device,non_blocking=True)

        Y = Y.long().to(args.device,non_blocking=True)
        if args.lap_pos_enc:
            batch_lap_pos_enc = g.ndata['lap_pos_enc']
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(args.device,non_blocking=True)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            g.ndata['lap_pos_enc'] = batch_lap_pos_enc * sign_flip.unsqueeze(0)

        logits = model(g,full_g)
        loss = loss_fn(logits, Y)
        train_losses.append(loss)

        loss = loss/args.grad_sum
        loss.backward()
        if (i_batch + 1) % args.grad_sum == 0  or i_batch == len(train_dataloader) - 1:
            optimizer.step()
            model.zero_grad()

        if args.ngpu >= 1:
            dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
            loss /= float(dist.get_world_size()) # get all loss value 
        loss = loss.data*args.grad_sum 
        if args.lr_decay:
            scheduler.step()
    return model,train_losses,optimizer,scheduler
def getToyKey(train_keys):

    """get toy dataset for test"""

    train_keys_toy_d = []
    train_keys_toy_a = []
    
    max_all = 600
    for key in train_keys:
        if '_active_' in key:
            train_keys_toy_a.append(key)
        if '_active_' not in key:
            train_keys_toy_d.append(key)

    if len(train_keys_toy_a) == 0 or len(train_keys_toy_d) == 0:
        return None

    return train_keys_toy_a[:300] + train_keys_toy_d[:(max_all-300)]
def getTestedPro(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            lines = []
            for line in f.readlines():
                if 'actions' in line:
                    lines.append(line)
        lines = [line.split('\t')[0] for line in lines]
        return lines
    else:
        return []
def getEF(model,args,test_path,save_path,debug,batch_size,loss_fn,rates = 0.01,flag = '',prot_split_flag = '_'):
        """calculate EF of test dataset, since dataset have 102/81 proteins ,so we need to calculate EF of each protein one by one!"""
        save_file = save_path + '/EF_test' + flag
        tested_pros = getTestedPro(save_file)
        test_keys = [key for key in os.listdir(test_path) if '.' not in key]
        pros = defaultdict(list)
        for key in test_keys:
            key_split = key.split(prot_split_flag)
 
            if '_active' in key:
                pros[key_split[0]].insert(0,os.path.join(test_path ,key))
            else:
                ''' all positive label sample will be place in head of list'''
                pros[key_split[0]].append(os.path.join(test_path ,key))


        EFs = []
        st = time.time()
        if type(rates) is not list:
                rates = list([rates])
        rate_str = ''
        for rate in rates:
            rate_str += str(rate)+ '\t'
        for pro in pros.keys():
            try :
                if pro in tested_pros:
                    if args.ngpu >= 1:
                        dist.barrier()
                    print('this pro :  %s  is tested'%pro)
                    continue
                test_keys_pro = pros[pro]
                if len(test_keys_pro) == 0:
                    if args.ngpu >= 1:
                        dist.barrier()
                    continue
                test_dataset = ESDataset(test_keys_pro,args, test_path,debug)
                val_sampler = SequentialDistributedSampler(test_dataset,args.batch_size) if args.ngpu >= 1 else None
                test_dataloader = DataLoaderX(test_dataset, batch_size = batch_size, \
                shuffle=False, num_workers = 8, collate_fn=test_dataset.collate,pin_memory = True,sampler = val_sampler)

                test_losses,test_true,test_pred = evaluator(model,test_dataloader,loss_fn,args,val_sampler)

                if args.ngpu >= 1:
                    dist.barrier()
                if args.local_rank == 0:
                    test_auroc,BEDROC,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(test_true,test_pred)
                    test_losses = torch.mean(torch.tensor(test_losses,dtype=torch.float)).data.cpu().numpy()
                    Y_sum = 0
                    for key in test_keys_pro:
                        key_split = key.split('_')
                        if '_active' in key:
                            Y_sum += 1
                    actions = int(Y_sum)
                    action_rate = actions/len(test_keys_pro)

                    EF = []
                    hits_list = []
                    for rate in rates:
                        ''' cal different rates of EF'''
                        find_limit = int(len(test_keys_pro)*rate)
                        _,indices = torch.sort(torch.tensor(test_pred),descending = True)
                        hits = torch.sum(indices[:find_limit] < actions)
                        EF.append((hits/find_limit)/action_rate)
                        hits_list.append(hits)

                    
                    EF_str = '['
                    hits_str = '['
                    for ef,hits in zip(EF,hits_list):
                        EF_str += '%.3f'%ef+'\t'
                        hits_str += ' %d '%hits
                    EF_str += ']'
                    hits_str += ']'
                    end = time.time()
                    with open(save_file,'a') as f:
                        f.write(pro+ '\t'+'actions: '+str(actions)+ '\t' + 'actions_rate: '+str(action_rate)+ '\t' + 'hits: '+ hits_str +'\t'+'loss:' + str(test_losses)+'\n'\
                            +'EF:'+rate_str+ '\t'+'test_auroc'+ '\t' + 'BEDROC' + '\t'+'test_adjust_logauroc'+ '\t'+'test_auprc'+ '\t'+'test_balanced_acc'+ '\t'+'test_acc'+ '\t'+'test_precision'+ '\t'+'test_sensitity'+ '\t'+'test_specifity'+ '\t'+'test_f1' +'\t' +'time'+ '\n')
                        f.write( EF_str + '\t'+str(test_auroc)+ '\t' + str(BEDROC) + '\t'+str(test_adjust_logauroc)+ '\t'+str(test_auprc)+ '\t'+str(test_balanced_acc)+ '\t'+str(test_acc)+ '\t'+str(test_precision)+ '\t'+str(test_sensitity)+ '\t'+str(test_specifity)+ '\t'+str(test_f1) +'\t'+ str(end-st)+ '\n')
                        f.close()
                    EFs.append(EF)
            except:
                print(pro,':skip for some bug')
                if args.ngpu >= 1:
                    dist.barrier()
                continue
            if args.ngpu >= 1:
                dist.barrier()
        if args.local_rank == 0:
            EFs = list(np.sum(np.array(EFs),axis=0)/len(EFs))
            EFs_str = '['
            for ef in EFs:
                EFs_str += str(ef)+'\t'
            EFs_str += ']'
            args_dict = vars(args)
            with open(save_file,'a') as f:
                    f.write( 'average EF for different EF_rate:' + EFs_str +'\n')
                    for item in args_dict.keys():
                        f.write(item + ' : '+str(args_dict[item]) + '\n')
                    f.close()
        if args.ngpu >= 1:
            # Keeping processes in sync
            dist.barrier()
def getNumPose(test_keys,nums = 5):

    """get the first nums pose for each ligand to prediction"""

    ligands = defaultdict(list)
    for key in test_keys:
        key_split = key.split('_')
        ligand_name = '_'.join(key_split[-2].split('-')[:-1])
        ligands[ligand_name].append(key)
    result = []
    for ligand_name in ligands.keys():
        ligands[ligand_name].sort(key = lambda x : int(x.split('_')[-2].split('-')[-1]),reverse=False)
        result += ligands[ligand_name][:nums]
    return result
def getIdxPose(test_keys,idx = 0):
    """"get the idx pose for each ligand to prediction"""
    ligands = defaultdict(list)


    for key in test_keys:
        key_split = key.split('_')
        ligand_name = '_'.join(key_split[-2].split('-')[:-1])
        ligands[ligand_name].append(key)
    result = []
    for ligand_name in ligands.keys():

        ligands[ligand_name].sort(key = lambda x : int(x.split('_')[-2].split('-')[-1]),reverse=False)
        if idx < len(ligands[ligand_name]):
            result.append(ligands[ligand_name][idx]) 
        else:
            result.append(ligands[ligand_name][-1]) 
    return result
def getEFMultiPose(model,args,test_path,save_path,debug,batch_size,loss_fn,rates = 0.01,flag = '',pose_num = 5,idx_style = False):
        """calulate EF for multi pose complex"""
        save_file = save_path + '/EF_test_multi_pose' + '_{}_'.format(pose_num) + flag
        test_keys = os.listdir(test_path)
        # for multi pose complex, get pose_num poses to cal EF
        tested_pros = getTestedPro(save_file)
        if idx_style:
            test_keys = getIdxPose(test_keys,idx = pose_num)
        else:
            test_keys = getNumPose(test_keys,nums = pose_num) 
 
        pros = defaultdict(list)
        for key in test_keys:
            key_split = key.split('_')
            pros[key_split[0]].append(os.path.join(test_path , key))
        EFs = []
        st = time.time()
        if type(rates) is not list:
                rates = list([rates])
        rate_str = ''
        for rate in rates:
            rate_str += str(rate)+ '\t'
        for pro in pros.keys():
            try :
                if pro in tested_pros:
                    if args.ngpu >= 1:
                        dist.barrier()
                    print('this pro :  %s  is tested'%pro)
                    continue
                test_keys_pro = pros[pro]
                if test_keys_pro is None:
                    if args.ngpu >= 1:
                        dist.barrier()
                    continue
                print('protein keys num ',len(test_keys_pro))

                test_dataset = ESDataset(test_keys_pro,args, test_path,debug)
                val_sampler = SequentialDistributedSampler(test_dataset,args.batch_size) if args.ngpu >= 1 else None
                test_dataloader = DataLoaderX(test_dataset, batch_size = batch_size, \
                shuffle=False, num_workers = 8, collate_fn=test_dataset.collate,pin_memory = True,sampler = val_sampler)
                test_losses,test_true,test_pred = evaluator(model,test_dataloader,loss_fn,args,val_sampler)

                if args.ngpu >= 1:
                    dist.barrier()
                if args.local_rank == 0:
 
                    test_auroc,BEDROC,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(test_true,test_pred)
                    test_losses = torch.mean(torch.tensor(test_losses,dtype=torch.float)).data.cpu().numpy()
                    Y_sum = 0
                    # multi pose 
                    # get max logits for every ligand
                    key_logits = defaultdict(list)
                    for pred,key in zip(test_pred,test_keys_pro):
                        new_key = '_'.join(key.split('/')[-1].split('_')[:-2] + key.split('/')[-1].split('_')[-2].split('-')[:-1])
                        key_logits[new_key].append(pred)
                    new_keys = list(key_logits.keys())
                    max_pose_logits = [max(logits) for logits in  list(key_logits.values())]

                    test_keys_pro = []
                    test_pred = []
                    for key,logit in zip(new_keys,max_pose_logits):
                        key_split = key.split('_') 
                        if 'actives' in key_split:
                            test_keys_pro.insert(0,key)
                            test_pred.insert(0,logit)
                            Y_sum += 1
                        else:
                            ''' all positive label sample will be place in head of list'''
                            test_keys_pro.append(key)
                            test_pred.append(logit)

                    actions = int(Y_sum)
                    action_rate = actions/len(test_keys_pro)
                    
                    EF = []
                    hits_list = []
                    for rate in rates:
                        find_limit = int(len(test_keys_pro)*rate)
                        _,indices = torch.sort(torch.tensor(test_pred),descending = True)
                        hits = torch.sum(indices[:find_limit] < actions)
                        EF.append((hits/find_limit)/action_rate)
                        hits_list.append(hits)
                    
                    EF_str = '['
                    hits_str = '['
                    for ef,hits in zip(EF,hits_list):
                        EF_str += '%.3f'%ef+'\t'
                        hits_str += ' %d '%hits
                    EF_str += ']'
                    hits_str += ']'
                    end = time.time()
                    with open(save_file,'a') as f:
                        f.write(pro+ '\t'+'actions: '+str(actions)+ '\t' + 'actions_rate: '+str(action_rate)+ '\t' + 'hits: '+ hits_str +'\t'+'loss:' + str(test_losses)+'\n'\
                            +'EF:'+rate_str+ '\t'+'test_auroc'+ '\t' + 'BEDROC' + '\t'+'test_adjust_logauroc'+ '\t'+'test_auprc'+ '\t'+'test_balanced_acc'+ '\t'+'test_acc'+ '\t'+'test_precision'+ '\t'+'test_sensitity'+ '\t'+'test_specifity'+ '\t'+'test_f1' +'\t' +'time'+ '\n')
                        f.write( EF_str + '\t'+str(test_auroc)+ '\t' + str(BEDROC) + '\t'+str(test_adjust_logauroc)+ '\t'+str(test_auprc)+ '\t'+str(test_balanced_acc)+ '\t'+str(test_acc)+ '\t'+str(test_precision)+ '\t'+str(test_sensitity)+ '\t'+str(test_specifity)+ '\t'+str(test_f1) +'\t'+ str(end-st)+ '\n')
                        f.close()
                    EFs.append(EF)
            except:
                print(pro,':skip for some bug')
                if args.ngpu >= 1:
                    dist.barrier()
                continue
            if args.ngpu >= 1:
                dist.barrier()
        if args.local_rank == 0:
            EFs = list(np.sum(np.array(EFs),axis=0)/len(EFs))
            EFs_str = '['
            for ef in EFs:
                EFs_str += str(ef)+'\t'
            EFs_str += ']'
            args_dict = vars(args)
            with open(save_file,'a') as f:
                    f.write( 'average EF for different EF_rate:' + EFs_str +'\n')
                    for item in args_dict.keys():
                        f.write(item + ' : '+str(args_dict[item]) + '\n')
                    f.close()
        if args.ngpu >= 1:
            dist.barrier()
def getEF_from_MSE(model,args,test_path,save_path,device,debug,batch_size,A2_limit,loss_fn,rates = 0.01):
        """cal EF for regression model if you want to training a regression model, you can use this function to cal EF"""
        
        save_file = save_path + '/EF_test'
        test_keys = [key for key in os.listdir(test_path) if '.' not in key]
        pros = defaultdict(list)
        for key in test_keys:
            key_split = key.split('_')
            if 'active' in key_split:
                pros[key_split[0]].insert(0,key)
            else:
                pros[key_split[0]].append(key)

        EFs = []
        st = time.time()
        if type(rates) is not list:
                rates = list([rates])
        rate_str = ''
        for rate in rates:
            rate_str += str(rate)+ '\t'
        for pro in pros.keys():
            try :

                test_keys_pro = pros[pro]
                if test_keys_pro is None:
                    continue
                test_dataset = ESDataset(test_keys_pro,args, test_path,debug)
                test_dataloader = DataLoader(test_dataset, batch_size = batch_size, \
                shuffle=False, num_workers = args.num_workers, collate_fn=test_dataset.collate)
                test_losses,test_true,test_pred = evaluator(model,test_dataloader,loss_fn,args)
                test_auroc,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(test_true,test_pred)
                test_losses = np.mean(np.array(test_losses))
                # print(test_losses)
                Y_sum = 0
                for key in test_keys_pro:
                    key_split = key.split('_')
                    if 'active' in key_split:
                        Y_sum += 1
                actions = int(Y_sum)
                action_rate = actions/len(test_keys_pro)
                test_pred = np.concatenate(np.array(test_pred), 0)
                EF = []
                hits_list = []
                for rate in rates:
                    find_limit = int(len(test_keys_pro)*rate)
                    _,indices = torch.sort(torch.tensor(test_pred),descending = True)
                    hits = torch.sum(indices[:find_limit] < actions)
                    EF.append((hits/find_limit)/action_rate)
                    hits_list.append(hits)
                
                EF_str = '['
                hits_str = '['
                for ef,hits in zip(EF,hits_list):
                    EF_str += '%.3f'%ef+'\t'
                    hits_str += ' %d '%hits
                EF_str += ']'
                hits_str += ']'
                end = time.time()
                with open(save_file,'a') as f:
                    f.write(pro+ '\t'+'actions: '+str(actions)+ '\t' + 'actions_rate: '+str(action_rate)+ '\t' + 'hits: '+ hits_str +'\n'\
                        +'EF:'+rate_str)
                    f.write( EF_str)
                    f.close()
                EFs.append(EF)
            except:
                print(pro,':skip for some bug')
                continue
        EFs = list(np.sum(np.array(EFs),axis=0)/len(EFs))
        EFs_str = '['
        for ef in EFs:
            EFs_str += str(ef)+'\t'
        EFs_str += ']'
        args_dict = vars(args)
        with open(save_file,'a') as f:
                f.write( 'average EF for different EF_rate:' + EFs_str +'\n')
                for item in args_dict.keys():
                    f.write(item + ' : '+str(args_dict[item]) + '\n')
                f.close()
from collections import defaultdict
import numpy as np
import pickle

def get_train_val_keys(keys):
    train_keys = keys
    pro_dict = defaultdict(list)
    for key in train_keys:
        pro = key.split('_')[0]
        pro_dict[pro].append(key)
    pro_list = list(pro_dict.keys())
    indices = np.arange(len(pro_list))
    np.random.shuffle(indices)
    train_num = int(len(indices)*0.8)
    count = 0
    train_list = []
    val_list = []
    for i in indices:
        count +=1
        if count < train_num:
            train_list += pro_dict[pro_list[i]]
        else:
            val_list +=  pro_dict[pro_list[i]]
    return train_list,val_list
def get_dataloader(args,train_keys,val_keys,val_shuffle=False):
    """"
    docstring:
        get dataloader for train and validation
    input:
        train_keys: list of train keys
            train file paths
        val_keys: list of validation keys
            validation file paths

    output: dataloader for train and validation
        (train_dataloader,val_dataloader)
    """
    train_dataset = ESDataset(train_keys,args, args.data_path,args.debug)
    val_dataset = ESDataset(val_keys,args, args.data_path,args.debug)
   
    if args.sampler:

        num_train_chembl = len([0 for k in train_keys if '_active' in k])
        num_train_decoy = len([0 for k in train_keys if '_active' not in k])
        train_weights = [1/num_train_chembl if '_active' in k else 1/num_train_decoy for k in train_keys]
        train_sampler = DTISampler(train_weights, len(train_weights), replacement=True)                     
        train_dataloader = DataLoader(train_dataset, args.batch_size, \
            shuffle=False,num_workers = args.num_workers, collate_fn=train_dataset.collate,\
            sampler = train_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, args.batch_size, \
            shuffle=True, num_workers = args.num_workers, collate_fn=train_dataset.collate)
    val_dataloader = DataLoader(val_dataset, args.batch_size, \
        shuffle=val_shuffle, num_workers = args.num_workers, collate_fn=val_dataset.collate)
    return train_dataloader,val_dataloader

def write_log_head(args,log_path,model,train_keys,val_keys):
    """a function to write the head of log file at the beginning of training"""
    args_dict = vars(args)
    with open(log_path,'w')as f:
        f.write(f'Number of train data: {len(train_keys)}' +'\n'+ f'Number of val data: {len(val_keys)}' + '\n')
        f.write(f'number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}' +'\n')
        for item in args_dict.keys():
            f.write(item + ' : '+str(args_dict[item]) + '\n')
        f.write('epoch'+'\t'+'train_loss'+'\t'+'val_loss'+'\t'+'test_loss' #'\t'+'train_auroc'+ '\t'+'train_adjust_logauroc'+ '\t'+'train_auprc'+ '\t'+'train_balanced_acc'+ '\t'+'train_acc'+ '\t'+'train_precision'+ '\t'+'train_sensitity'+ '\t'+'train_specifity'+ '\t'+'train_f1'+ '\t'\
        + '\t' + 'test_auroc'+ '\t' + 'BEDROC' + '\t'+'test_adjust_logauroc'+ '\t'+'test_auprc'+ '\t'+'test_balanced_acc'+ '\t'+'test_acc'+ '\t'+'test_precision'+ '\t'+'test_sensitity'+ '\t'+'test_specifity'+ '\t'+'test_f1' +'\t' +'time'+ '\n')
        f.close()
def save_model(model,optimizer,args,epoch,save_path,mode = 'best'):
    """a function to save model"""
    best_name = save_path + '/save_{}_model'.format(mode)+'.pt'
    if args.debug:
        best_name = save_path + '/save_{}_model_debug'.format(mode)+'.pt'

    torch.save({'model':model.module.state_dict() if isinstance(model,nn.parallel.DistributedDataParallel) else model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch}, best_name)

def shuffle_train_keys(train_keys):
    """shuffle train keys by protein"""
    sample_dict = defaultdict(list)
    for i in train_keys:
        key = i.split('/')[-1].split('_')[0]
        sample_dict[key].append(i)
    keys = list(sample_dict.keys())
    np.random.shuffle(keys)
    new_keys = []
    batch_sizes = []

    for i,key in enumerate(keys):
        temp = sample_dict[key]
        np.random.shuffle(temp)
        new_keys += temp
        batch_sizes.append(len(temp))
    return new_keys,batch_sizes
