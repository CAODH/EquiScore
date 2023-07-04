import time
import utils.utils as utils
from utils.utils import *
from utils.loss_utils import *
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
from torch.multiprocessing import Process
def run(local_rank,args,*more_args,**kwargs):
    args.local_rank = local_rank
    torch.distributed.init_process_group(backend="nccl",init_method='env://',rank = args.local_rank,world_size = args.ngpu)  # initial distribution training，'nccl'mode
    torch.cuda.set_device(args.local_rank) 
    seed_torch(seed = args.seed + args.local_rank)
    args_dict = vars(args)
    if args.FP:
        args.N_atom_features = 39
    else:
        args.N_atom_features = 28
    model = EquiScore(args) if args.model == 'EquiScore' else None

    args.device = args.local_rank
    best_name = args.save_model
    model_name = best_name.split('/')[-1]
    save_path = best_name.replace(model_name,'')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.test_path = os.path.join(args.test_path,args.test_name)
    model = utils.initialize_model(model, args.device,args, load_save_file = best_name )[0]
    if args.loss_fn == 'bce_loss':
        loss_fn = nn.BCELoss().to(args.device)# 
    elif args.loss_fn == 'focal_loss':
        loss_fn = FocalLoss().to(args.device)
    elif args.loss_fn == 'cross_entry':
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smothing).to(args.device)
    elif args.loss_fn == 'mse_loss':
        loss_fn = nn.MSELoss().to(args.device)
    elif args.loss_fn == 'poly_loss_ce':
        loss_fn = PolyLoss_CE(epsilon = args.eps).to(args.device)
    elif args.loss_fn == 'poly_loss_fl':
        loss_fn = PolyLoss_FL(epsilon=args.eps,gamma = 2.0).to(args.device)
    else:
        raise ValueError('not support this loss : %s'%args.loss_fn)

    # if you docking result have different pose number, you can use the following functions

    """
    1. if one compound has only one pose, you can use the following function
    """
    if args.test_mode=='one_pose':
        getEF(model,args,args.test_path,save_path,args.debug,args.batch_size,loss_fn,args.EF_rates,\
              flag = args.test_flag + '{}_'.format(model_name)+ args.test_name,prot_split_flag = '_')
    if args.test_mode == 'multi_pose':
        """
        2. if one compound has multi pose, you can use the following function
        pose_num to select top pose_num poses for test 
        if idx_style is true, the pose_num is the pose index,only one pose on the index be used to test, else the pose_num is the pose number  
        
        """
        getEFMultiPose(model,args,args.test_path,save_path,args.debug,args.batch_size,loss_fn,rates = args.EF_rates,\
                       flag = args.test_flag + '{}_'.format(model_name) +  args.test_name,pose_num = args.pose_num,idx_style=args.idx_style)
if '__main__' == __name__:
    from torch import distributed as dist
    import torch.multiprocessing as mp
    from utils.dist_utils import *
    from utils.parsing import parse_train_args
    args = parse_train_args()
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
    processes = []
    for rank in range(world_size):
        p = Process(target=run, args=(rank, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()