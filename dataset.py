
import lmdb
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
# from dataset_utils import get_atom_graphformer_feature
import utils 
import numpy as np
import torch
import random
from scipy.spatial import distance_matrix
import pickle
import dgl
import dgl.data
from ifp_construct import get_nonBond_pair
from dataset_utils import *
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

class ESDataset(Dataset):

    def __init__(self, keys,args, data_dir,debug):
        super(ESDataset, self).__init__()
        self.keys = keys
        self.data_dir = data_dir
        self.debug = debug
        self.args = args
        self.graphs = []
        if not args.test:
            # load data from LMDB database
            env = lmdb.open(args.lmdb_cache, map_size=int(1e12), max_dbs=2, readonly=True)
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

        samples = list(filter(lambda  x : x is not None,samples))
        g,full_g,Y = map(list, zip(*samples))
        batch_g = dgl.batch(g)
        batch_full_g = dgl.batch(full_g)
        Y = torch.tensor(Y).long()
        return batch_g, batch_full_g,Y
    def __getitem__(self, idx):
        key = self.keys[idx]
        if not self.args.test:
            g,Y= pickle.loads(self.txn.get(key.encode(), db=self.graph_db))
        else:
            try:
                g,Y = self._GetGraph(key,self.args)
            except:
                return None
        # get covalent bond based edges
        a,b = g.edges()
        # construct geometric distance based graph 
        dm_all = distance_matrix(g.ndata['coors'].numpy(),g.ndata['coors'].numpy())#g.ndata['coors'].matmul(g.ndata['coors'].T)
        edges_g = np.concatenate([a.reshape(-1,1).numpy(),b.reshape(-1,1).numpy()],axis = 1)
        src,dst = np.where(dm_all < self.args.threshold)
        # add covalent bond based edges and remove duplicated edges
        edges_full = np.concatenate([src.reshape(-1,1),dst.reshape(-1,1)],axis = 1)
        edges_full = np.unique(np.concatenate([edges_full,edges_g],axis = 0),axis = 0)
        full_g = dgl.graph((edges_full[:,0],edges_full[:,1]))
        full_g.ndata['coors'] = g.ndata['coors'] 
        g.ndata.pop('coors') 

        return g,full_g,Y

    @staticmethod
    def _GetGraph(key,args):
        ''' 
        construct structual graph based on covalent bond and non-bond interaction and save to LMDB database for speed up data loading
  '''
        try:
            try:
                with open(key, 'rb') as f:
                    m1,m2= pickle.load(f)
            except:
                with open(key, 'rb') as f:
                    m1,m2,atompairs,iter_types= pickle.load(f)
        except:
            return None
        n1,d1,adj1 = get_mol_info(m1)
        n2,d2,adj2 = get_mol_info(m2)

        H1 = np.concatenate([get_atom_graphformer_feature(m1,FP = args.FP) ,np.array([0]).reshape(1,-1).repeat(n1,axis = 0)],axis=1)
        H2 = np.concatenate([get_atom_graphformer_feature(m2,FP = args.FP) ,np.array([1]).reshape(1,-1).repeat(n2,axis = 0)],axis=1)
        if args.virtual_aromatic_atom:
            adj1,H1,d1,n1 = add_atom_to_mol(m1,adj1,H1,d1,n1)
            adj2,H2,d2,n2 = add_atom_to_mol(m2,adj2,H2,d2,n2)
        H = torch.from_numpy(np.concatenate([H1, H2], 0))
        # get covalent bond edges
        agg_adj1 = np.zeros((n1+n2, n1+n2))
        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2
        # get non-bond interactions based edges
        if args.fingerprintEdge:
            # slowly when trainging from raw data ,so we save the result in disk,for next time use
            if 'inter_types' not in vars().keys() and 'atompairs' not in vars().keys():
                try:
                    atompairs,iter_types = get_nonBond_pair(m1,m2)
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
        agg_adj1 = torch.from_numpy(agg_adj1)
        adj_graph_1 = np.copy(agg_adj1)
        pocket = (m1,m2)
        item_1 = mol2graph(pocket,H,args,adj = adj_graph_1,n1 = n1,n2 = n2,\
            dm = (d1,d2) )
        g = preprocess_item(item_1, args,adj_graph_1)

        g.ndata['coors'] = torch.from_numpy(np.concatenate([d1,d2],axis=0))
        valid = torch.zeros((n1+n2,))
        # set readout feature for task layer
        if args.pred_mode == 'ligand':
            valid[:n1] = 1
        elif args.pred_mode == 'protein':
            valid[n1:] = 1
        else:
            raise ValueError(f'not support this mode : {args.pred_mode} plz check the args')
        if args.loss_fn == 'mse_loss':
            Y = -float(key.split('-')[1].split('_')[0])
        else:
            if '_active' in key.split('/')[-1]:
                Y = 1 
            else:
                Y = 0 
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

    from parsing import parse_train_args
    args = parse_train_args()

    # create lmdb database for map data and key,this step can help speed up training! Also ,can can skip this step too.
    
    env = lmdb.open(args.lmdb_cache, map_size=int(1e12), max_dbs=1)
    # create lmdb database
    dgl_graph_db = env.open_db('data'.encode())
    # read all data file path from pkl file  
    ''' Attention you should change contain all data path in test_keys when you process data to LMDB database ,also \
        you can just specity a file path directly rather than passing it via args vatriable!'''
    with open (args.test_keys, 'rb') as fp:
        val_keys = pickle.load(fp)
    keys =  val_keys 
    ################save processed data into database and then you can index data by key##########################
    def saveDB(key):
        with env.begin(write=True) as txn:
            try:
                g,y = ESDataset._GetGraph(key,args)
                # print(type(g))
                txn.put(key.encode(), pickle.dumps((g,y)), db = dgl_graph_db)
            except:
                print('file: {} is not a valid file!'.format(key))
    all_keys = len(keys)
    with Pool(processes = 32) as pool:
        list(pool.imap(saveDB, keys))
    print('save done!')
    env.close()




        

       