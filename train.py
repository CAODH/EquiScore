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



from torch.multiprocessing import Process
def run(local_rank,args,*more_args,**kwargs):
    # seed_everything()
    # rank = torch.distributed.get_rank()
    args.local_rank = local_rank
    # print('in run args',local_rank,args.local_rank)
    torch.distributed.init_process_group(backend="nccl",init_method='env://',rank = args.local_rank,world_size = args.ngpu)  # 并行训练初始化，'nccl'模式
    torch.cuda.set_device(args.local_rank) 
    # rank = torch.distributed.get_rank()
    # print('seed seed to single process')
    seed_torch(seed = args.seed + args.local_rank)
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


    model = gnn_edge(args) if args.gnn_edge else gnn(args) 

    print ('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    if args.hot_start:
        model ,opt_dict,epoch_start= utils.initialize_model(model, args.device,args,args.save_model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # print('opt_dict: ',opt_dict)
        optimizer.load_state_dict(opt_dict)
        # print('optimizer: ',optimizer)
    else:

        model = utils.initialize_model(model, args.device,args)
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
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = SequentialDistributedSampler(val_dataset,args.batch_size)
    if args.sampler:

        num_train_chembl = len([0 for k in train_keys if '_active' in k])
        num_train_decoy = len([0 for k in train_keys if '_active' not in k])
        train_weights = [1/num_train_chembl if '_active' in k else 1/num_train_decoy for k in train_keys]
        train_sampler = DTISampler(train_weights, len(train_weights), replacement=True)                     
        train_dataloader = DataLoaderX(train_dataset, args.batch_size, \
            shuffle=False, num_workers = args.num_workers, collate_fn=collate_fn,\
            sampler = train_sampler,pin_memory=True,drop_last = True)#动态采样
    else:
        train_dataloader = DataLoaderX(train_dataset, args.batch_size,sampler = train_sampler, \
            shuffle=False, num_workers = args.num_workers,drop_last = True, collate_fn=collate_fn,pin_memory=True)
    val_dataloader = DataLoaderX(val_dataset, args.batch_size, sampler = val_sampler,\
        shuffle=False, num_workers = args.num_workers, drop_last = True, collate_fn=collate_fn,pin_memory=True)
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
        train_sampler.set_epoch(epoch)
        model,train_losses,optimizer = train(model,args,optimizer,loss_fn,train_dataloader,auxiliary_loss)
        # print('train func len of train loss',len(train_losses))
        val_losses,val_true,val_pred = evaluator(model,val_dataloader,loss_fn,args,val_sampler)
        if args.lr_decay:
            scheduler.step()
        # dist.barrier()
        # if epoch >-1:
        if local_rank == 0:
            train_losses = np.mean(np.array(train_losses))
            # test_losses = np.mean(np.array(test_losses))
            val_losses = np.mean(np.array(val_losses))
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
        dist.barrier()
    print('training done!')
    
if '__main__' == __name__:
    from torch import distributed as dist
    import torch.multiprocessing as mp
    from dist_utils import *
    def get_args_from_json(json_file_path, args_dict):
        import json
        summary_filename = json_file_path
        with open(summary_filename) as f:
            summary_dict = json.load(fp=f)
        for key in summary_dict.keys():
            args_dict[key] = summary_dict[key]
        return args_dict
    parser = argparse.ArgumentParser(description='json param')
    parser.add_argument('--local_rank', default=-1, type=int) 
    parser.add_argument("--json_path", help="file path of param", type=str, \
        default='/home/caoduanhua/score_function/GNN/GNN_graphformer_pyg/train_keys/config_files/gnn_edge_3d_pos_dist.json')
    args = parser.parse_args()
    local_rank = args.local_rank
    # label_smoothing# temp_args = parser.parse_args()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(args_dict['json_path'], args_dict)
    args = argparse.Namespace(**args)
    # 下面这个参数需要加上，torch内部调用多进程时，会使用该参数，对每个gpu进程而言，其local_rank都是不同的；
    args.local_rank = local_rank
    
    # parser.add_argument('--local_rank', default=-1, type=int)  
    # print(args.local_rank)
    # torch.cuda.set_device(args.local_rank)  # 设置gpu编号为local_rank;此句也可能看出local_rank的值是什么
    # torch.nn.parallel.DistributedDataParallel()
    # print (args)
    # torch.distributed.init_process_group(backend="nccl",init_method='env://',rank = local_rank,world_size = args.ngpu)  # 并行训练初始化，'nccl'模式
    # print('world_size', torch.distributed.get_world_size()) # 打印当前进程数
    if args.ngpu>0:
        cmd = get_available_gpu(num_gpu=args.ngpu, min_memory=30000, sample=3, nitro_restriction=False, verbose=True)
        if cmd[-1] == ',':
            os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES']=cmd
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    # os.environ[
    #     "TORCH_DISTRIBUTED_DEBUG"
    # ] = "DETAIL"  # set to DETAIL for runtime logging.
    
    # mp.spawn(run, nprocs=args.ngpu, args=(args,)) slow than lanch 
    from torch.multiprocessing import Process
    world_size = args.ngpu
    processes = []
    for rank in range(world_size):
        p = Process(target=run, args=(rank, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


