import pickle
import time
import numpy as np
import utils.utils as utils
from utils.utils import *
from utils.loss_utils import *
# from dataset_utils import *
from utils.dist_utils import *
from dataset.dataset import *
import torch.nn as nn
import torch
import time
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import argparse
import time
from torch.utils.data import DataLoader          
from prefetch_generator import BackgroundGenerator
from model.equiscore import EquiScore
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())    
now = time.localtime()
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print (s)
os.chdir(os.path.abspath(os.path.dirname(__file__)))
def run(local_rank,args):
    args.local_rank = local_rank
    torch.distributed.init_process_group(backend="nccl",init_method='env://',rank = args.local_rank,world_size = args.ngpu)  # multi gpus training，'nccl' mode
    torch.cuda.set_device(args.local_rank) 
    seed_torch(seed = args.seed + args.local_rank)
    # use attentiveFP feature or not
    if args.FP:
        args.N_atom_features = 39
    else:
        args.N_atom_features = 28
    num_epochs = args.epoch
    lr = args.lr
    save_dir = args.save_dir
    train_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    # make save dir if it doesn't exist
    if args.hot_start:
        if os.path.exists(args.save_model):
            best_name = args.save_model
            model_name = best_name.split('/')[-1]
            save_path = best_name.replace(model_name,'')
        else:
            raise ValueError('save_model is not a valid file check it again!')
    else:
        save_path = os.path.join(save_dir,args.model,train_time)

    if not os.path.exists(save_path):
        os.system('mkdir -p ' + save_path)
    log_path = save_path+'/logs' 

    #read data. data is stored in format of dictionary. Each key has information about protein-ligand complex.

    if args.train_val_mode == 'uniport_cluster':
        with open (args.train_keys, 'rb') as fp:
            train_keys = pickle.load(fp)
        with open (args.val_keys, 'rb') as fp:
            val_keys = pickle.load(fp)
        with open (args.test_keys, 'rb') as fp:
            test_keys = pickle.load(fp)
    else:
        raise 'not implement this split mode,check the config file plz'

        # /pass
    if local_rank == 0:
        print (f'Number of train data: {len(train_keys)}')
        print (f'Number of val data: {len(val_keys)}')
        print (f'Number of test data: {len(test_keys)}')

    model = EquiScore(args) if args.model == 'EquiScore' else None
    print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    args.device = args.local_rank
    if args.hot_start:
        model ,opt_dict,epoch_start= utils.initialize_model(model, args.device,args,args.save_model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        optimizer.load_state_dict(opt_dict)

    else:
        model = utils.initialize_model(model, args.device,args)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        epoch_start = 0
        write_log_head(args,log_path,model,train_keys,val_keys)
    # dataset processing
    train_dataset = ESDataset(train_keys,args, args.data_path,args.debug)#keys,args, data_dir,debug
    val_dataset = ESDataset(val_keys,args, args.data_path,args.debug)
    test_dataset = ESDataset(test_keys,args, args.data_path,args.debug) 
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.ngpu >= 1 else None
    val_sampler = SequentialDistributedSampler(val_dataset,args.batch_size) if args.ngpu >= 1 else None
    test_sampler = SequentialDistributedSampler(test_dataset,args.batch_size) if args.ngpu >= 1 else None
#    use sampler to balance the training data or not 
    if args.sampler:
        num_train_chembl = len([0 for k in train_keys if '_active' in k])
        num_train_decoy = len([0 for k in train_keys if '_active' not in k])
        train_weights = [1/num_train_chembl if '_active' in k else 1/num_train_decoy for k in train_keys]
        train_sampler = DTISampler(train_weights, len(train_weights), replacement=True)                     
        train_dataloader = DataLoaderX(train_dataset, args.batch_size, \
            shuffle=False, num_workers = args.num_workers, collate_fn=train_dataset.collate,prefetch_factor = 4,\
            sampler = train_sampler,pin_memory=True,drop_last = True) #dynamic sampler
    else:
        train_dataloader = DataLoaderX(train_dataset, args.batch_size, sampler = train_sampler,\
            shuffle=False, num_workers = args.num_workers, collate_fn=train_dataset.collate,pin_memory=True,prefetch_factor = 4)
    val_dataloader = DataLoaderX(val_dataset, args.batch_size, sampler=val_sampler,\
        shuffle=False, num_workers = args.num_workers, collate_fn=val_dataset.collate,pin_memory=True,prefetch_factor = 4)
    test_dataloader = DataLoaderX(test_dataset, args.batch_size, sampler=test_sampler,\
        shuffle=False, num_workers = args.num_workers, collate_fn=test_dataset.collate,pin_memory=True,prefetch_factor = 4) 

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.max_lr,pct_start=args.pct_start,\
         steps_per_epoch=len(train_dataloader), epochs=args.epoch,last_epoch = -1 if len(train_dataloader)*epoch_start == 0 else len(train_dataloader)*epoch_start )
    #loss function ,in this paper just use cross entropy loss but you can try focal loss too!
    if args.loss_fn == 'bce_loss':
        loss_fn = nn.BCELoss().to(args.device,non_blocking=True)# 
    elif args.loss_fn == 'focal_loss':
        loss_fn = FocalLoss().to(args.device,non_blocking=True)
    elif args.loss_fn == 'cross_entry':
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smothing).to(args.device,non_blocking=True)
    elif args.loss_fn == 'mse_loss':
        loss_fn = nn.MSELoss().to(args.device,non_blocking=True)
    elif args.loss_fn == 'poly_loss_ce':
        loss_fn = PolyLoss_CE(epsilon = args.eps).to(args.device,non_blocking=True)
    elif args.loss_fn == 'poly_loss_fl':
        loss_fn = PolyLoss_FL(epsilon=args.eps,gamma = 2.0).to(args.device,non_blocking=True)
    else:
        raise ValueError('not support this loss : %s'%args.loss_fn)

    best_loss = 1000000000
    best_f1 = -1
    counter = 0
    for epoch in range(epoch_start,num_epochs):
        st = time.time()
        #collect losses of each iteration
        if args.ngpu >= 1:
            train_sampler.set_epoch(epoch) 
        model,train_losses,optimizer,scheduler = train(model,args,optimizer,loss_fn,train_dataloader,scheduler)
        if args.ngpu >= 1:
            dist.barrier() 
        val_losses,val_true,val_pred = evaluator(model,val_dataloader,loss_fn,args,val_sampler)
        if args.ngpu >= 1:
            dist.barrier() 
        if local_rank == 0:
            test_losses = 0.0
            train_losses = torch.mean(torch.tensor(train_losses,dtype=torch.float)).data.cpu().numpy()
            val_losses = torch.mean(torch.tensor(val_losses,dtype=torch.float)).data.cpu().numpy()
            if args.loss_fn == 'mse_loss':
                end = time.time()
                with open(log_path,'a') as f:
                    f.write(str(epoch)+ '\t'+str(train_losses)+ '\t'+str(val_losses)+ '\t'+str(test_losses) + str(end-st)+ '\n')
                    f.close()
            else:
                test_auroc,BEDROC,test_adjust_logauroc,test_auprc,test_balanced_acc,test_acc,test_precision,test_sensitity,test_specifity,test_f1 = get_metrics(val_true,val_pred)
                end = time.time()
                with open(log_path,'a') as f:
                    f.write(str(epoch)+ '\t'+str(train_losses)+ '\t'+str(val_losses)+ '\t'+str(test_losses)\
                    + '\t'+str(test_auroc)+ '\t'+str(BEDROC) + '\t'+str(test_adjust_logauroc)+ '\t'+str(test_auprc)+ '\t'+str(test_balanced_acc)+ '\t'+str(test_acc)+ '\t'+str(test_precision)+ '\t'+str(test_sensitity)+ '\t'+str(test_specifity)+ '\t'+str(test_f1) +'\t'\
                    + str(end-st)+ '\n')
                    f.close()
            counter +=1 
            if val_losses < best_loss:
                best_loss = val_losses
                counter = 0
                save_model(model,optimizer,args,epoch,save_path,mode = 'best')
            if test_f1 > best_f1:
                best_f1 = test_f1
                counter = 0
                save_model(model,optimizer,args,epoch,save_path,mode = 'best_f1')
            if counter > args.patience:
                save_model(model,optimizer,args,epoch,save_path,mode = 'early_stop')
                print('model early stop !')
                break
            if epoch == num_epochs-1:
                save_model(model,optimizer,args,epoch,save_path,mode = 'end')
        if args.ngpu >= 1:
            dist.barrier() 
    if args.ngpu >= 1:
        dist.barrier() 
    print('training done!')
    
if '__main__' == __name__:
    from torch import distributed as dist
    import torch.multiprocessing as mp
    from utils.dist_utils import *
    from utils.parsing import parse_train_args
    # get args from parsering function
    args = parse_train_args()
    # set gpu to use
    if args.ngpu>0:
        cmd = get_available_gpu(num_gpu=args.ngpu, min_memory=28000, sample=3, nitro_restriction=False, verbose=True)
        if cmd[-1] == ',':
            os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES']=cmd
    os.environ["MASTER_ADDR"] = args.MASTER_ADDR
    os.environ["MASTER_PORT"] = args.MASTER_PORT
    from torch.multiprocessing import Process
    world_size = args.ngpu

    # use multiprocess to train

    processes = []
    for rank in range(world_size):
        p = Process(target=run, args=(rank, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


