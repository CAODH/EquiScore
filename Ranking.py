import pickle
from dist_utils import *
import time
import utils
from utils import *
import glob
import torch
import time
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
# os.envirment[]
import argparse
import time
from torch.utils.data import DataLoader          
# from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from equiscore import EquiScore
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())                            
from dataset import Dataset, collate_fn, DTISampler
now = time.localtime()
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
print (s)
os.chdir(os.path.abspath(os.path.dirname(__file__)))

from torch.multiprocessing import Process
def run(local_rank,args,*more_args,**kwargs):

    args.local_rank = local_rank
    torch.distributed.init_process_group(backend="nccl",init_method='env://',rank = args.local_rank,world_size = args.ngpu)  # 并行训练初始化，'nccl'模式
    torch.cuda.set_device(args.local_rank) 

    seed_torch(seed = args.seed + args.local_rank)

    args_dict = vars(args)
    if args.FP:
        args.N_atom_features = 39
    else:
        args.N_atom_features = 28

    model = EquiScore(args) if args.gnn_model == 'EquiScore' else None
   
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = args.local_rank
    best_name = args.save_model
    model_name = best_name.split('/')[-1]
    save_path = best_name.replace(model_name,'')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.test_path = os.path.join(args.test_path,args.test_name)
    test_keys_pro = glob.glob(args.test_path + '/*')
    test_dataset = Dataset(test_keys_pro,args, args.test_path,args.debug)
    test_sampler = SequentialDistributedSampler(test_dataset,args.batch_size) if args.ngpu >= 1 else None
    test_dataloader = DataLoaderX(test_dataset, batch_size = args.batch_size, \
    shuffle=False, num_workers = 8, collate_fn=test_dataset.collate,pin_memory = True,sampler = test_sampler)
    #model, args.device,args,args.save_model
    model = utils.initialize_model(model, args.device,args, load_save_file = best_name )[0]

    model.eval()
    with torch.no_grad():
        test_pred = []
        for i_batch, (g,full_g,Y) in enumerate(test_dataloader):
            # time_s = time.time()
            # print('g,full_g',g,full_g,Y)
            model.zero_grad()
            g = g.to(args.local_rank,non_blocking=True)
            full_g = full_g.to(args.local_rank,non_blocking=True)
            # Y = Y.long().to(args.local_rank,non_blocking=True)
            pred = model(g,full_g)
            if pred.dim()==2:
                pred = torch.softmax(pred,dim = -1)[:,1]
            pred = pred if args.loss_fn == 'auc_loss' else pred
            test_pred.append(pred.data) if args.ngpu >= 1 else test_pred.append(pred.data)
            # print(test_pred)
        # gather ngpu result to single tensor
        if args.ngpu >= 1:
            test_pred = distributed_concat(torch.concat(test_pred, dim=0), 
                                            len(test_sampler.dataset)).cpu().numpy()
        else:
            test_pred = torch.concat(test_pred, dim=0).cpu().numpy()
    if args.ngpu >= 1:
        with open(args.pred_save_path,'wb') as f:
            pickle.dump((test_keys_pro,test_pred),f)
    # return test_losses,test_true,test_pred
# import copy
    # getEF(model,args,args.test_path,save_path,args.device,args.debug,args.batch_size,args.A2_limit,loss_fn,args.EF_rates,flag = '_' + args.test_name,prot_split_flag = '_')
    # getEFMultiPose(model,args,args.test_path,save_path,args.debug,args.batch_size,loss_fn,rates = args.EF_rates,flag = '_RTMScore_testdata' + args.test_name,pose_num = 3)
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
        default='/home/caoduanhua/score_function/GNN/config_keys_results/new_data_train_keys/config_rank/gnn_edge_3d_pos_screen_dgl_FP_pose_enhanced_challenge_cross_10_threshold_55_large_rank_fep.json')
    args = parser.parse_args()
    local_rank = args.local_rank
    # label_smoothing# temp_args = parser.parse_args()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(args_dict['json_path'], args_dict)
    args = argparse.Namespace(**args)
    # 下面这个参数需要加上，torch内部调用多进程时，会使用该参数，对每个gpu进程而言，其local_rank都是不同的；
    args.local_rank = local_rank
    if args.ngpu>0:
        cmd = get_available_gpu(num_gpu=args.ngpu, min_memory=30000, sample=3, nitro_restriction=False, verbose=True)
        if cmd[-1] == ',':
            os.environ['CUDA_VISIBLE_DEVICES']=cmd[:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES']=cmd
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"

    from torch.multiprocessing import Process
    world_size = args.ngpu
    processes = []
    for rank in range(world_size):
        p = Process(target=run, args=(rank, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()