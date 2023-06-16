import time
import utils
# from utils import *
import torch.nn as nn
import torch
import time
import os
# from dist_utils import SequentialDistributedSampler
import glob
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import argparse
import time
from torch.utils.data import DataLoader          
from prefetch_generator import BackgroundGenerator
from dataset import ESDataset
import pickle
from equiscore import EquiScore
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
    torch.distributed.init_process_group(backend="nccl",init_method='env://',rank = args.local_rank,world_size = args.ngpu) #initial distribution training，'nccl'mode
    torch.cuda.set_device(args.local_rank) 
    seed_torch(seed = args.seed + args.local_rank)
    args_dict = vars(args)
    if args.FP:
        args.N_atom_features = 39
    else:
        args.N_atom_features = 28
    model =EquiScore(args) if args.model == 'EquiScore' else None
    args.device = args.local_rank
    best_name = args.save_model
    model_name = best_name.split('/')[-1]
    save_path = best_name.replace(model_name,'')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.test_path = os.path.join(args.test_path,args.test_name)
    test_keys_pro = glob.glob(args.test_path + '/*')
    test_dataset = ESDataset(test_keys_pro,args, args.test_path,args.debug)
    test_sampler = SequentialDistributedSampler(test_dataset,args.batch_size) if args.ngpu >= 1 else None
    test_dataloader = DataLoaderX(test_dataset, batch_size = args.batch_size, \
    shuffle=False, num_workers = 8, collate_fn=test_dataset.collate,pin_memory = True,sampler = test_sampler)
    model = utils.initialize_model(model, args.device,args, load_save_file = best_name )[0]
    model.eval()
    with torch.no_grad():
        test_pred = []
        for i_batch, (g,full_g,Y) in enumerate(test_dataloader):
            model.zero_grad()
            g = g.to(args.local_rank,non_blocking=True)
            full_g = full_g.to(args.local_rank,non_blocking=True)
            pred = model(g,full_g)
            if pred.dim()==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu >= 1 else test_pred.append(pred.data)
        # gather ngpu result to single tensor
        if args.ngpu >= 1:
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        else:
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    if args.ngpu >= 1:
        os.makedirs(os.path.dirname(args.pred_save_path),exist_ok=True)
        with open(args.pred_save_path,'wb') as f:
            pickle.dump((test_keys_pro,test_pred),f)
if '__main__' == __name__:
    '''distribution training'''
    from torch import distributed as dist
    import torch.multiprocessing as mp
    from dist_utils import *
    from parsing import parse_train_args
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