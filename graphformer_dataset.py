from graphformer_utils import *
# import graphformer_utils
# from graphformer_utils import get_atom_graphformer_feature
import os
# from utils import *
import lmdb
from  tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import utils 
import numpy as np
import torch
import random
from rdkit import Chem
from scipy.spatial import distance_matrix
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import pickle
import dgl
import dgl.data
from fp_construct import getNonBondPair
# import 
random.seed(42)
class DTISampler(Sampler):
    def __init__(self, weights, num_samples, replacement=True):
        weights = np.array(weights)/np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
    def __iter__(self):
        #return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())
        retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights) 
        return iter(retval.tolist())
    def __len__(self):
        return self.num_samples
def get_atom_feature(m, is_ligand=True,FP = False):
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        if FP:
            H.append(utils.atom_feature_attentive_FP(m.GetAtomWithIdx(i),
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=True))
        else:
            H.append(utils.atom_feature(m, i, None, None))
    H = np.array(H)        
    if is_ligand:
        H = np.concatenate([H, np.zeros((n,len(H[-1])))], 1)
    else:
        H = np.concatenate([np.zeros((n,len(H[-1]))), H], 1)
    return H
#重写GNN_DTID的dataset
class graphformerDataset(Dataset):

    def __init__(self, keys,args, data_dir,debug):
        self.keys = keys
        self.data_dir = data_dir
        self.debug = debug
        self.args = args
        self.graphs = []
        if not args.test:
            env = lmdb.open(f'/home/caoduanhua/score_function/data/lmdbs/pose_challenge_cross_10', map_size=int(1e12), max_dbs=2, readonly=True)
            self.graph_db = env.open_db('data'.encode()) # graph data base
            self.txn = env.begin(buffers=True,write=False)
        else:
            pass

    def __len__(self):
        if self.debug:
            return 30000
        return len(self.keys)
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        ''' collate function for building graph dataloader'''
        # samples = list(filter(lambda  x : x is not None,samples))
        g,full_g,Y = map(list, zip(*samples))

        batch_g = dgl.batch(g)
        batch_full_g = dgl.batch(full_g)
        Y = torch.tensor(Y).long()
        return batch_g, batch_full_g,Y
    def __getitem__(self, idx):
        #idx = 0
        # time_s = time.time()
        # time_strst = time.time()
        key = self.keys[idx]
        if not self.args.test:
            g,Y= pickle.loads(self.txn.get(key.encode(), db=self.graph_db))
        else:
            g,Y = graphformerDataset._GetGraph(key,self.args)

        a,b = g.edges()
        dm_all = distance_matrix(g.ndata['coors'].numpy(),g.ndata['coors'].numpy())#g.ndata['coors'].matmul(g.ndata['coors'].T)
        edges_g = np.concatenate([a.reshape(-1,1).numpy(),b.reshape(-1,1).numpy()],axis = 1)
        src,dst = np.where(dm_all < self.args.threshold) # add sparse edges and remove duplicated edges
        edges_full = np.concatenate([src.reshape(-1,1),dst.reshape(-1,1)],axis = 1)
        edges_full = np.unique(np.concatenate([edges_full,edges_g],axis = 0),axis = 0)

        full_g = dgl.graph((edges_full[:,0],edges_full[:,1]))
        # time_g_full = time.time()
        # print('load g_full grapg:',time.time() - time_g_full)

        # full_g.edata['adj2'] = agg_adj2.view(-1,1).contiguous()
        return g,full_g,Y

    @staticmethod
    def _GetGraph(key,args):
        
        try:
            try:
                with open(key, 'rb') as f:
                    m1,m2= pickle.load(f)
                # f.close()
            except:
                with open(key, 'rb') as f:
                    m1,m2,atompairs,iter_types= pickle.load(f)
                f.close()
        except:
            print('file: {} is not a valid file！'.format(key))
            return None
        n1,d1,adj1 = utils.get_mol_info(m1)
        n2,d2,adj2 = utils.get_mol_info(m2)

        H1 = np.concatenate([get_atom_graphformer_feature(m1,FP = args.FP) ,np.array([0]).reshape(1,-1).repeat(n1,axis = 0)],axis=1)
        H2 = np.concatenate([get_atom_graphformer_feature(m2,FP = args.FP) ,np.array([1]).reshape(1,-1).repeat(n2,axis = 0)],axis=1)
        # print('max,min atom fea BEFORE',np.max(H1),np.min(H1))
        if args.virtual_aromatic_atom:
            adj1,H1,d1,n1 = utils.add_atom_to_mol(m1,adj1,H1,d1,n1)
            # print( adj1,H1,d1,n1)
            adj2,H2,d2,n2 = utils.add_atom_to_mol(m2,adj2,H2,d2,n2)
            # print( adj2,H2,d2,n2)
        # print('max,min atom fea after',np.max(H1),np.min(H1))
        H = torch.from_numpy(np.concatenate([H1, H2], 0))
        agg_adj1 = np.zeros((n1+n2, n1+n2))
        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2
        agg_adj2 = np.copy(agg_adj1)
        # add fp edge 
        if args.fingerprintEdge:
            # 边跑边处理太慢了， 预处理再加载
            if 'inter_types' not in vars().keys() and 'atompairs' not in vars().keys():
                try:
                    atompairs,iter_types = getNonBondPair(m1,m2)
                except:
                    atompairs,iter_types = [],[]
                    # print(key)
                with open(key,'wb') as f:
                    pickle.dump((m1,m2,atompairs,iter_types),f)
                f.close()
                
            if len(atompairs) > 0:
                temp_fp= np.array(atompairs)
                u,v = list(temp_fp[:,0]) +  list((n1+ temp_fp[:,1])),list((n1+ temp_fp[:,1])) + list(temp_fp[:,0])
                agg_adj1[u,v] = 1
        dm = distance_matrix(d1,d2)
        dm_all = distance_matrix(np.concatenate([d1,d2],axis=0),np.concatenate([d1,d2],axis=0))
        if args.only_dis_adj2:
            agg_adj2 = dm_all
        elif args.dis_adj2_with_adj1:# have som troubles to fix
            # dm_all = distance_matrix(np.concatenate([d1,d2],axis=0),np.concatenate([d1,d2],axis=0))
            agg_adj2 = dm_all+ agg_adj1
        else:
            agg_adj2[:n1,n1:] = np.copy(dm)
            agg_adj2[n1:,:n1] = np.copy(np.transpose(dm))
        agg_adj1 = torch.from_numpy(agg_adj1)
        agg_adj2 = torch.from_numpy(agg_adj2)
        # full_g.edata['adj1'] = agg_adj1.view(-1,1).contiguous()
        size = (n1,n2)
        # dm = np.where(dm <5.0 ,1,0)
        adj_graph_1 = np.copy(agg_adj1)
        # adj_graph_2 = np.copy(agg_adj1)

        pocket = (m1,m2)
        item_1 = mol2graph(pocket,H,args,adj = adj_graph_1,n1 = n1,n2 = n2,\
            dm = (d1,d2) )
        # item_time = time.time()
        # print('item_time get item_1:',time.time()-time_s)
        g = preprocess_item(item_1, args,adj_graph_1)

        g.ndata['coors'] = torch.from_numpy(np.concatenate([d1,d2],axis=0))
        valid = torch.zeros((n1+n2,))
        if args.pred_mode == 'ligand':
            valid[:n1] = 1
        elif args.pred_mode == 'protein':
            valid[n1:] = 1
        elif args.pred_mode == 'supernode':
            valid = torch.zeros((n1+n2+1,))
            assert args.supernode == True,'plz setup your supernode in graph!'
            # assert len() == n1+n2+1 ,'no super node added to pocket graph check the args plz'
            valid[-1] = 1
        else:
            raise ValueError(f'not support this mode : {self.args.pred_mode} plz check the args')
        if args.loss_fn == 'mse_loss':
            Y = -float(key.split('-')[1].split('_')[0])
        else:
            if '_active' in key.split('/')[-1]:
                Y = 1 
                if args.add_logk_reg:
                    try:
                        value = self.logk_match[key.split('/')[-1].split('_')[0]]
                    except:
                        # pdbscreen active not have value
                        value = 0
                else:
                    value = 0
            else:
                Y =  0 
                value = 0
        g.ndata['V'] = valid.float().reshape(-1,1)
        return g,Y


    
if __name__ == "__main__":
    import lmdb
    import argparse
    from rdkit import RDLogger
    from dgl import save_graphs, load_graphs
    import dgl
    from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED
    import time
    from multiprocessing import Pool, cpu_count
    import os
    RDLogger.DisableLog('rdApp.*')
    # from train import get_args_from_json
    # from train
    parser = argparse.ArgumentParser(description='json param')
    parser.add_argument("--json_path", help="file path of param", type=str, \
        default='/home/caoduanhua/score_function/GNN/GNN_graphformer_pyg/new_data_train_keys/config_files/gnn_edge_3d_pos_screen_dgl_FP_pose_enhanced_challenge_cross_10.json')

    # label_smoothing# temp_args = parser.parse_args()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(args_dict['json_path'], args_dict)
    args = argparse.Namespace(**args)
    # print (args)
    from tqdm import tqdm
    from functools import partial
    # 创建数据库文件
    env = lmdb.open(f'/home/caoduanhua/score_function/data/lmdbs/pose_challenge_cross_10', map_size=int(1e12), max_dbs=1)
    # 创建对应的数据库
    # mol_pocket_inter_idx_type = env.open_db('mol_pocket_interIdx_type'.encode())
    dgl_graph_db = env.open_db('data'.encode())
    # 把数据写入到LMDB中
    with open (args.train_keys, 'rb') as fp:
        train_keys = pickle.load(fp)
    with open (args.val_keys, 'rb') as fp:
        val_keys = pickle.load(fp)

    keys =  val_keys + train_keys
    ##########################################
    def saveDB(key):
        with env.begin(write=True) as txn:
            try:
                g,y = graphformerDataset._GetGraph(key,args)
                # print(type(g))
                txn.put(key.encode(), pickle.dumps((g,y)), db = dgl_graph_db)
            except:
                print('file: {} is not a valid file!'.format(key))

    all_keys = len(keys)
    # pbar = tqdm(keys)
    with Pool(processes = 32) as pool:
        list(pool.imap(saveDB, keys))
        # list(tqdm(pool.imap(saveDB, keys), total=all_keys))
    print('save done!')
    env.close()
    ########################################需要实现多线程或者多进程加多线程



        

       