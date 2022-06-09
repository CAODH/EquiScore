import numpy as np
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from scipy import sparse
import os.path
import time
import torch.nn as nn
import math
from dataset import *
import random
from collections import defaultdict
# from graphformer_dataset import *
# from ase import Atoms, Atom
from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve
from sklearn.metrics import accuracy_score,auc,balanced_accuracy_score#g,l;x,y;g,l
from sklearn.metrics import recall_score,precision_score,precision_recall_curve#g,l;g,l;g,p
from sklearn.metrics import confusion_matrix,f1_score#g,l;g,l
#from rdkit.Contrib.SA_Score.sascorer import calculateScore
#from rdkit.Contrib.SA_Score.sascorer
#import deepchem as dc
# import  multiprocessing as mp
# mp.set_spawn_method("spawn")
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())                            
from graphformer_dataset import graphformerDataset,  DTISampler
N_atom_features = 28
import glob
import pickle
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import rdkit
def get_args_from_json(json_file_path, args_dict):
    import json
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)
    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]
    return args_dict

def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    empty = []
    for i in range(8):
        command = 'nvidia-smi -i '+str(i)+' | grep "No running" | wc -l'
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        #print('nvidia-smi -i '+str(i)+' | grep "No running" | wc -l > empty_gpu_check')
        if int(output)==1:
            empty.append(i)
    if len(empty)<ngpus:
        print ('avaliable gpus are less than required')
        exit(-1)
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    return cmd


def initialize_model(model, device, args,load_save_file=False,init_classifer = True):
    for param in model.parameters():
        if param.dim() == 1:
            continue
            nn.init.constant_(param, 0)
        else:
            #nn.init.normal(param, 0.0, 0.15)
            nn.init.xavier_normal_(param)
            

    if load_save_file:
        state_dict = torch.load(load_save_file,map_location = 'cpu')
        model_dict = state_dict['model']
        # model_dict.pop('deta')
        model_state_dict = model.state_dict()
        # model_dict = {k:v for k,v in model_dict.items() if 'FC' not in k}# gengxin
        model_state_dict.update(model_dict)
        model.load_state_dict(model_state_dict) 
        
        optimizer =state_dict['optimizer']
        epoch = state_dict['epoch']
        print('load save model!')
    # model.to(device)
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

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_feature(m, atom_i, i_donor, i_acceptor):

    atom = m.GetAtomWithIdx(atom_i)
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (10, 6, 5, 6, 1) --> total 28
def get_aromatic_rings(mol:rdkit.Chem.Mol) -> list:
    ''' return aromaticatoms rings'''
    aromaticity_atom_id_set = set()
    rings = []
    for atom in mol.GetAromaticAtoms():
        aromaticity_atom_id_set.add(atom.GetIdx())
    # get ring info 
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        ring_id_set = set(ring)
        # check atom in this ring is aromaticity
        if ring_id_set <= aromaticity_atom_id_set:
            rings.append(list(ring))
    return rings
def add_atom_to_mol(mol,adj,H,d,n):
    '''docstring: 
    add virtual aromatic atom feature/adj/3d_positions to raw data
    '''
    assert len(adj) == len(H),'adj nums not equal to nodes'
    rings = get_aromatic_rings(mol)
    num_aromatic = len(rings)
    
    h,b = adj.shape
    # print(num_aromatic,h,b)
    # print(d.shape,H.shape)
    all_zeros = np.zeros((num_aromatic+h,num_aromatic+b))
    #add all zeros vector to bottom and right

    all_zeros[:h,:b] = adj
    for i,ring in enumerate(rings):
        all_zeros[h+i,:][ring] = 1
        all_zeros[:,h+i][ring] = 1
        all_zeros[h+i,:][h+i] = 1
        d = np.concatenate([d,np.mean(d[ring],axis = 0,keepdims=True)],axis = 0)
        H  = np.concatenate([H,np.array([0]*(H.shape[1]))[np.newaxis]],axis = 0)
    assert len(all_zeros) == len(H),'adj nums not equal to nodes'
    return all_zeros,H,d,n+num_aromatic
def get_mol_info(m1):
    n1 = m1.GetNumAtoms()
    c1 = m1.GetConformers()[0]
    d1 = np.array(c1.GetPositions())
    adj1 = GetAdjacencyMatrix(m1)+np.eye(n1)
    return n1,d1,adj1
def atom_feature_attentive_FP(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=True):
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:
        results = one_of_k_encoding_unk(
          atom.GetSymbol(),
          [
            'B',
            'C',
            'N',
            'O',
            'F',
            'Si',
            'P',
            'S',
            'Cl',
            'As',
            'Se',
            'Br',
            'Te',
            'I',
            'At',
            'other'
          ]) + one_of_k_encoding(atom.GetDegree(),
                                 [0, 1, 2, 3, 4, 5]) + \
                  [int(atom.GetFormalCharge()), int(atom.GetNumRadicalElectrons())] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2,'other'
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results) #size all 39 [16 6 1 1 6 1 5 1 2 ]
def get_logauc(fp, tp, min_fp=0.001, adjusted=False):
    
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
def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_metrics(train_true,train_pred):
    try:
        train_pred = np.concatenate(np.array(train_pred,dtype=object), 0).astype(np.float)
        train_true = np.concatenate(np.array(train_true,dtype=object), 0).astype(np.long)
    except:
        pass
    # print(train_pred,train_true)
    train_pred_label = np.where(train_pred > 0.5,1,0).astype(np.long)

    tn, fp, fn, tp = confusion_matrix(train_true,train_pred_label).ravel()
    train_auroc = roc_auc_score(train_true, train_pred) 
    train_acc = accuracy_score(train_true,train_pred_label)
    train_precision = precision_score(train_true,train_pred_label)
    train_sensitity = tp/(tp + fn)#recall_score(train_true,train_pred_label)
    train_specifity = tn/(fp+tn)
    ps,rs,_ = precision_recall_curve(train_true,train_pred)
    train_auprc = auc(rs,ps)
    train_f1 = f1_score(train_true,train_pred_label)
    train_balanced_acc = balanced_accuracy_score(train_true,train_pred_label)
    fp,tp,_ = roc_curve(train_true,train_pred)
    train_adjusted_logauroc = get_logauc(fp,tp)
    return train_auroc,train_adjusted_logauroc,train_auprc,train_balanced_acc,train_acc,train_precision,train_sensitity,train_specifity,train_f1

#by caoduanhua
def random_split(train_keys, split_ratio=0.9, seed=0, shuffle=True):
    dataset_size = len(train_keys)
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(split_ratio * dataset_size)
    train_idx, valid_idx = indices[:split], indices[split:]
    return [train_keys[i] for i in train_idx], [train_keys[i] for i in valid_idx]
def data_to_device(sample,device):

        data_flag = []
        data = []
        # print('sample',sample)
        for i in sample.get_att():
            if type(i) is torch.Tensor:
                data.append(i.to(device))
                data_flag.append(1)
            else:
                data_flag.append(None)
        return data_flag,data
def average_gradients(model):  ##每个gpu上的梯度求平均
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data,op = torch.distributed.ReduceOp.SUM)
            param.grad.data /= size

from dist_utils import *
def evaluator(model,loader,loss_fn,args,test_sampler):
    model.eval()
    with torch.no_grad():
        test_losses,test_true,test_pred = [], [],[]
        for i_batch, (g,full_g,Y) in enumerate(loader):
            # time_s = time.time()
            # print('g,full_g',g,full_g,Y)
            model.zero_grad()
            g = g.to(args.local_rank)
            full_g = full_g.to(args.local_rank)
            Y = Y.long().to(args.local_rank)
            pred = model(g,full_g)
            loss = loss_fn(pred ,Y) 
            # print(loss)
            if args.ngpu > 1:
            # dist.barrier() 
                dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
                loss /= float(dist.get_world_size()) # get all loss value 
            #collect loss, true label and predicted label
            test_losses.append(loss.data)
            if args.ngpu > 1:
                test_true.append(Y.data)
            else:
                test_true.append(Y.data)
            # print(test_true)
            if pred.dim()==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu > 1 else test_pred.append(pred.data)
            # print(test_pred)
        # gather ngpu result to single tensor
        if args.ngpu > 1:
            test_true = distributed_concat(torch.concat(test_true, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        
        else:
            test_true = torch.concat(test_true, dim=0).cpu().numpy()
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    return test_losses,test_true,test_pred
import copy
def train(model,args,optimizer,loss_fn,train_dataloader,auxiliary_loss,scheduler):
# 加入辅助函数和r_drop 方式
        #collect losses of each iteration
    train_losses = [] 
    # train_true = []
    # train_pred = []
    model.train()
    for i_batch, (g,full_g,Y) in enumerate(train_dataloader):
        g = g.to(args.device)
        full_g = full_g.to(args.device)
        if args.lap_pos_enc:

            batch_lap_pos_enc = g.ndata['lap_pos_enc']
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(args.device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            g.ndata['lap_pos_enc'] = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        pred = model(g,full_g)
        loss = loss_fn(pred, Y.long().to(pred.device))

        if args.grad_sum:
            loss = loss/6
            loss.backward()
            if (i_batch + 1) % 6 == 0  or i_batch == len(train_dataloader) - 1:
                optimizer.step()
                model.zero_grad()
                # print('batch_loss:',np.mean(train_losses))
        else:
            loss.backward()
            optimizer.step()
            model.zero_grad()
        if args.ngpu > 1:
            # dist.barrier() 
            dist.all_reduce(loss.data,op = torch.distributed.ReduceOp.SUM)
            loss /= float(dist.get_world_size()) # get all loss value 
        loss = loss.data*6 if args.grad_sum else loss.data
        train_losses.append(loss)
        scheduler.step()
    return model,train_losses,optimizer
def getToyKey(train_keys):
    train_keys_toy_d = []
    train_keys_toy_a = []
    
    max_all = 600
    for key in train_keys:
        
        if '_active_' in key:
            train_keys_toy_a.append(key)
            
       
        if '_active_' not in key:
            train_keys_toy_d.append(key)
    # print(train_keys_toy_a)
    # print(train_keys_toy_d)
    if len(train_keys_toy_a) == 0 or len(train_keys_toy_d) == 0:
        return None


    # train_keys_toy_a[:300] + train_keys_toy_d[:(max_all-300)]
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
def getEF(model,args,test_path,save_path,device,debug,batch_size,A2_limit,loss_fn,rates = 0.01,flag = ''):
        save_file = save_path + '/EF_test' + flag
        tested_pros = getTestedPro(save_file)
        test_keys = [key for key in os.listdir(test_path) if '.' not in key]
        pros = defaultdict(list)
        for key in test_keys:
            key_split = key.split('_')
            # if 'KAT2A' == key_split[0]:
            if 'active' in key_split:
                pros[key_split[0]].insert(0,os.path.join(test_path ,key))
            else:#阳性标签排在前面
                pros[key_split[0]].append(os.path.join(test_path ,key))

            #agg sme pro
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
                    if args.ngpu > 1:
                        dist.barrier()
                    print('this pro :  %s  is tested'%pro)
                    continue
                test_keys_pro = pros[pro]
                if test_keys_pro is None:
                    if args.ngpu > 1:
                        dist.barrier()
                    continue
                test_dataset = graphformerDataset(test_keys_pro,args, test_path,debug)
                val_sampler = SequentialDistributedSampler(test_dataset,args.batch_size) if args.ngpu > 1 else None
                test_dataloader = DataLoaderX(test_dataset, batch_size = batch_size, \
                shuffle=False, num_workers = 8, collate_fn=test_dataset.collate,pin_memory = True,sampler = val_sampler)

                test_losses,test_true,test_pred = evaluator(model,test_dataloader,loss_fn,args,val_sampler)

                if args.ngpu > 1:
                    dist.barrier()
                if args.local_rank == 0:
                    test_auroc,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(test_true,test_pred)
                    test_losses = torch.mean(torch.tensor(test_losses,dtype=torch.float)).data.cpu().numpy()
                    Y_sum = 0
                    for key in test_keys_pro:
                        key_split = key.split('_')
                        if 'active' in key_split:
                            Y_sum += 1
                    actions = int(Y_sum)
                    action_rate = actions/len(test_keys_pro)
                    #保存logits 进行下一步分析
                    # test_pred = np.concatenate(np.array(test_pred), 0)
                    EF = []
                    hits_list = []
                    for rate in rates:
                        find_limit = int(len(test_keys_pro)*rate)
                        # print(find_limit)
                        _,indices = torch.sort(torch.tensor(test_pred),descending = True)
                        hits = torch.sum(indices[:find_limit] < actions)
                        EF.append((hits/find_limit)/action_rate)
                        hits_list.append(hits)
                        # print(hits,actions,action_rate)
                    
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
                            +'EF:'+rate_str+ '\t'+'test_auroc'+ '\t'+'test_adjust_logauroc'+ '\t'+'test_auprc'+ '\t'+'test_balanced_acc'+ '\t'+'test_acc'+ '\t'+'test_precision'+ '\t'+'test_sensitity'+ '\t'+'test_specifity'+ '\t'+'test_f1' +'\t' +'time'+ '\n')
                        f.write( EF_str + '\t'+str(test_auroc)+ '\t'+str(test_adjust_logauroc)+ '\t'+str(test_auprc)+ '\t'+str(test_balanced_acc)+ '\t'+str(test_acc)+ '\t'+str(test_precision)+ '\t'+str(test_sensitity)+ '\t'+str(test_specifity)+ '\t'+str(test_f1) +'\t'+ str(end-st)+ '\n')
                        f.close()
                    EFs.append(EF)
            except:
                print(pro,':skip for some bug')
                if args.ngpu > 1:
                    dist.barrier()
                continue
            if args.ngpu > 1:
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
        if args.ngpu > 1:
            dist.barrier()
def getNumPose(test_keys,nums = 5):
    pros = defaultdict(list)
    for key in test_keys:
        key_split = key.split('_-')
        pros[key_split[0]].append(key)
    for key in pros.keys():
        pros[key].sort(key = lambda x : float(x.split('_-')[-1].replace('.sdf','')),reverse=True)
        pros[key] = pros[key][:nums]
    result = []
    for temp  in pros.values():
        result += temp
    return result
def getEFMultiPose(model,args,test_path,save_path,debug,batch_size,loss_fn,rates = 0.01,flag = '',pose_num = 5):
        save_file = save_path + '/EF_test_multi_pose' + flag
        test_keys = os.listdir(test_path)
        # 每个复合物提取固定比例的pose
        test_keys = getNumPose(test_keys,nums = pose_num)
        # print('tests nums',len(test_keys))
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
                test_keys_pro = pros[pro]
                if test_keys_pro is None:
                    if args.ngpu > 1:
                        dist.barrier()
                    continue

                test_dataset = graphformerDataset(test_keys_pro,args, test_path,debug)
                val_sampler = SequentialDistributedSampler(test_dataset,args.batch_size) if args.ngpu > 1 else None
                test_dataloader = DataLoaderX(test_dataset, batch_size = batch_size, \
                shuffle=False, num_workers = 8, collate_fn=test_dataset.collate,pin_memory = True,sampler = val_sampler)
                test_losses,test_true,test_pred = evaluator(model,test_dataloader,loss_fn,args,val_sampler)

                if args.ngpu > 1:
                    dist.barrier()
                if args.local_rank == 0:
                    test_auroc,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(test_true,test_pred)
                    test_losses = torch.mean(torch.tensor(test_losses,dtype=torch.float)).data.cpu().numpy()
                    Y_sum = 0
                    # multi pose 
                    # 一个ligand 一个名字，带有所有的概率，用来求概率最大值，然后，
                    # ligand_single_pose_with_max_logits = []
                    # new_keys = []
                    key_logits = defaultdict(list)
                    for pred,key in zip(test_pred,test_keys_pro):
                        new_key = '_'.join(key.split('_')[:-1])
                        key_logits[new_key].append(pred)
                    new_keys = list(key_logits.keys())
                    max_pose_logits = [max(logits) for logits in  list(key_logits.values())]

                    # test_keys_pro = new_keys
                    # test_pred = max_pose_logits
                    test_keys_pro = []
                    test_pred = []
                    for key,logit in zip(new_keys,max_pose_logits):
                        key_split = key.split('_') 
                        if 'active' in key_split:
                            test_keys_pro.insert(0,key)
                            test_pred.insert(0,logit)
                            Y_sum += 1
                        else:#阳性标签排在前面
                            test_keys_pro.append(key)
                            test_pred.append(logit)

                    actions = int(Y_sum)
                    action_rate = actions/len(test_keys_pro)
                    
                    #保存logits 进行下一步分析
                    # test_pred = np.concatenate(np.array(test_pred), 0)
                    EF = []
                    hits_list = []
                    for rate in rates:
                        find_limit = int(len(test_keys_pro)*rate)
                        # print(find_limit)
                        _,indices = torch.sort(torch.tensor(test_pred),descending = True)
                        hits = torch.sum(indices[:find_limit] < actions)
                        EF.append((hits/find_limit)/action_rate)
                        hits_list.append(hits)
                        # print(hits,actions,action_rate)
                    
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
                            +'EF:'+rate_str+ '\t'+'test_auroc'+ '\t'+'test_adjust_logauroc'+ '\t'+'test_auprc'+ '\t'+'test_balanced_acc'+ '\t'+'test_acc'+ '\t'+'test_precision'+ '\t'+'test_sensitity'+ '\t'+'test_specifity'+ '\t'+'test_f1' +'\t' +'time'+ '\n')
                        f.write( EF_str + '\t'+str(test_auroc)+ '\t'+str(test_adjust_logauroc)+ '\t'+str(test_auprc)+ '\t'+str(test_balanced_acc)+ '\t'+str(test_acc)+ '\t'+str(test_precision)+ '\t'+str(test_sensitity)+ '\t'+str(test_specifity)+ '\t'+str(test_f1) +'\t'+ str(end-st)+ '\n')
                        f.close()
                    EFs.append(EF)
            except:
                print(pro,':skip for some bug')
                if args.ngpu > 1:
                    dist.barrier()
                continue
            if args.ngpu > 1:
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
        if args.ngpu > 1:
            dist.barrier()
def getEF_from_MSE(model,args,test_path,save_path,device,debug,batch_size,A2_limit,loss_fn,rates = 0.01):

        
        save_file = save_path + '/EF_test'
        test_keys = [key for key in os.listdir(test_path) if '.' not in key]
        
        pros = defaultdict(list)
        for key in test_keys:
            key_split = key.split('_')
            if 'active' in key_split:
                pros[key_split[0]].insert(0,key)
            else:#阳性标签排在前面
                pros[key_split[0]].append(key)

            #agg sme pro
        EFs = []
        st = time.time()
        if type(rates) is not list:
                rates = list([rates])
        rate_str = ''
        for rate in rates:
            rate_str += str(rate)+ '\t'
        for pro in pros.keys():
            # with open(save_file,'a') as f:

            #     # f.write('')
            #     f.write(pro+ '\n'+'EF:'+rate_str+ '\t'+'test_auroc'+ '\t'+'test_adjust_logauroc'+ '\t'+'test_auprc'+ '\t'+'test_balanced_acc'+ '\t'+'test_acc'+ '\t'+'test_precision'+ '\t'+'test_sensitity'+ '\t'+'test_specifity'+ '\t'+'test_f1' +'\t' +'time'+ '\n')
            #     f.close()
            try :

                test_keys_pro = pros[pro]

                # test_keys_pro =  getToyKey(test_keys_pro)#just for test
                if test_keys_pro is None:
                    continue
                # print(len(test_keys_pro))

                test_dataset = graphformerDataset(test_keys_pro,args, test_path,debug)
                test_dataloader = DataLoader(test_dataset, batch_size = batch_size, \
                shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn)
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
    #保存logits 进行下一步分析
                
                test_pred = np.concatenate(np.array(test_pred), 0)
                # if args.save_logits:

                #     with open(save_path + '/pcba_{}_logits'.format(pro),'wb') as f:
                #         pickle.dump((test_pred,actions),f)
                #         f.close()
    
                EF = []
                hits_list = []
                for rate in rates:
                    find_limit = int(len(test_keys_pro)*rate)
                    # print(find_limit)
                    _,indices = torch.sort(torch.tensor(test_pred),descending = True)
                    hits = torch.sum(indices[:find_limit] < actions)
                    EF.append((hits/find_limit)/action_rate)
                    hits_list.append(hits)
                    # print(hits,actions,action_rate)
                
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
#focal losss 用于 imblanceed data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1).long()
        # target = target.i()
        # print(input.shape)
        logpt = F.log_softmax(input,dim = -1)
        # print(logpt.shape)
        # print(target.shape)
      
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def get_available_gpu(num_gpu=1, min_memory=1000, sample=3, nitro_restriction=True, verbose=True):
    '''
    :param num_gpu: number of GPU you want to use
    :param min_memory: minimum memory
    :param sample: number of sample
    :param nitro_restriction: if True then will not distribute the last GPU for you.
    :param verbose: verbose mode
    :return: str of best choices, e.x. '1, 2'
    '''
    sum = None
    for _ in range(sample):
        info = os.popen('nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv').read()
        # print(info)
        info = np.array([[id] + t.replace('%', '').replace('MiB','').split(',') for id, t in enumerate(info.split('\n')[1:-1])]).\
            astype(np.int)
        # print(info)
        sum = info + (sum if sum is not None else 0)
        # print(sum)
        time.sleep(0.2)
    avg = sum//sample
    # print(avg)

    if nitro_restriction:
        avg = avg[:-1]
    available = avg[np.where(avg[:,2] > min_memory)]  
    # print(available)  
    if len(available) < num_gpu:
        print ('avaliable gpus are less than required')
        exit(-1)
        # print()
    if available.shape[0] == 0:
        print('No GPU available')
        return ''
    select = ', '.join(available[np.argsort(available[:,1])[:num_gpu],0].astype(np.str).tolist())
    if verbose:
        print('Available GPU List')
        first_line = [['id', 'utilization.gpu(%)', 'memory.free(MiB)']]
        matrix = first_line + available.astype(np.int).tolist()
        s = [[str(e) for e in row] for row in matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
        print('Select id #' + select + ' for you.')
    return select

from collections import defaultdict
import numpy as np
import pickle

def get_train_val_keys(keys):
    train_keys = keys
    #按照蛋白来随机分
    # from collections import defaultdict
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
            # if len(pro_dict[pro_list[i]]) != 11:
            #     print(len(pro_dict[pro_list[i]]))
        else:
            val_list +=  pro_dict[pro_list[i]]
    return train_list,val_list
def get_dataloader(args,train_keys,val_keys,val_shuffle=False):
    train_dataset = graphformerDataset(train_keys,args, args.data_path,args.debug)#keys,args, data_dir,debug
    val_dataset = graphformerDataset(val_keys,args, args.data_path,args.debug)
    # test_dataset = graphformerDataset(test_keys,args, args.data_path,args.debug)
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
        shuffle=val_shuffle, num_workers = args.num_workers, collate_fn=collate_fn)
    return train_dataloader,val_dataloader

def write_log_head(args,log_path,model,train_keys,val_keys):
    args_dict = vars(args)
    with open(log_path,'w')as f:
        f.write(f'Number of train data: {len(train_keys)}' +'\n'+ f'Number of val data: {len(val_keys)}' + '\n')
        f.write(f'number of parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad)}' +'\n')
        for item in args_dict.keys():
            f.write(item + ' : '+str(args_dict[item]) + '\n')
        f.write('epoch'+'\t'+'train_loss'+'\t'+'val_loss'+'\t'+'test_loss' #'\t'+'train_auroc'+ '\t'+'train_adjust_logauroc'+ '\t'+'train_auprc'+ '\t'+'train_balanced_acc'+ '\t'+'train_acc'+ '\t'+'train_precision'+ '\t'+'train_sensitity'+ '\t'+'train_specifity'+ '\t'+'train_f1'+ '\t'\
        + '\t' + 'test_auroc'+ '\t'+'test_adjust_logauroc'+ '\t'+'test_auprc'+ '\t'+'test_balanced_acc'+ '\t'+'test_acc'+ '\t'+'test_precision'+ '\t'+'test_sensitity'+ '\t'+'test_specifity'+ '\t'+'test_f1' +'\t' +'time'+ '\n')
        f.close()
def save_model(model,optimizer,args,epoch,save_path,mode = 'best'):

    best_name = save_path + '/save_{}_model'.format(mode)+'.pt'
    if args.debug:
        best_name = save_path + '/save_{}_model_debug'.format(mode)+'.pt'

    torch.save({'model':model.module.state_dict() if isinstance(model,nn.parallel.DistributedDataParallel) else model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch}, best_name)


# 按照蛋白shuffle 数据
def shuffle_train_keys(train_keys):
    sample_dict = defaultdict(list)
    for i in train_keys:
        key = i.split('/')[-1].split('_')[0]
        sample_dict[key].append(i)
    keys = list(sample_dict.keys())
    # print(len(keys))
    np.random.shuffle(keys)
    new_keys = []
    batch_sizes = []

    for i,key in enumerate(keys):
        temp = sample_dict[key]
        np.random.shuffle(temp)
        new_keys += temp
        batch_sizes.append(len(temp))
    return new_keys,batch_sizes
#定义一个辅助的loss 用来惩罚阴性样本概率高于阳性样本

class auxiliary_loss(nn.Module):
    def __init__(self,args):
        super(auxiliary_loss,self).__init__()
        if args.deta_const:
            self.deta = 0.2
        else:
            self.deta = nn.Parameter(torch.Tensor([deta]).float())
    def forward(self,y_pred,labels):
        y_pred = y_pred.reshape(1,-1)
        labels = labels.reshape(1,-1)
        pos_num = torch.sum(labels)
        neg_num = len(labels)-pos_num
        if len(labels) > neg_num > 0:
            # print('y_pred:',y_pred)
            # print('labels:',labels.bool())
            pos_pred = y_pred[labels.bool()]
            neg_pred = y_pred[(1-labels).bool()]
            loss = self.deta*torch.sum(neg_pred - pos_pred.reshape(-1,1))/(pos_num*neg_num)
            # loss = Variable(loss, requires_grad=True)
        else: 
            loss = 0
        return loss 
from torch.nn.functional import one_hot
class PolyLoss_FL(torch.nn.Module):
    """
    Implementation of poly loss FOR FL.
    Refers to `PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions (ICLR 2022)
    """

    def __init__(self, num_classes=2, epsilon=1.0,gamma = 2.0):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.criterion = FocalLoss(gamma = 2.0,size_average = False)
        self.num_classes = num_classes
        self.gamma = gamma

    def forward(self, output, target):
        fl = self.criterion(output, target)
        pt = one_hot(target.long(), num_classes=self.num_classes) * self.softmax(output)
        return (fl + self.epsilon * torch.pow(1.0 - pt.sum(dim=-1),self.gamma + 1)).mean()


class PolyLoss_CE(torch.nn.Module):
    """
    Implementation of poly loss for CE.
    Refers to `PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions (ICLR 2022)
    """

    def __init__(self, num_classes=2, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes

    def forward(self, output, target):
        ce = self.criterion(output, target.long())
        pt = one_hot(target.long(), num_classes=self.num_classes) * self.softmax(output)

        return (ce + self.epsilon * (1.0 - pt.sum(dim=-1))).mean()