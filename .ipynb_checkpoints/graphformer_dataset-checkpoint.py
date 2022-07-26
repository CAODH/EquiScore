from graphformer_utils import *
# import graphformer_utils
# from graphformer_utils import get_atom_graphformer_feature
import os
# from utils import *

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
random.seed(0)
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
        if self.args.add_logk_reg:
            with open('/home/caoduanhua/score_function/data/general_refineset/datapro/logk_match.pkl','rb') as f:
                self.logk_match =  pickle.load(f)
            # self.logk_match = 
    def __len__(self):
        if self.debug:
            return 5680
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
        #idx = 0
        key = self.keys[idx]
        file_path = self.args.path_data_dir+'/'+key
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
        # pocket = Chem.CombineMols(m1,m2)
        # H1 = get_atom_graphformer_feature(m1,FP = self.args.FP)
        # H2 = get_atom_graphformer_feature(m2,FP = self.args.FP)
        H1 = np.concatenate([get_atom_graphformer_feature(m1,FP = self.args.FP) ,np.array([0]).reshape(1,-1).repeat(n1,axis = 0)],axis=1)
        H2 = np.concatenate([get_atom_graphformer_feature(m2,FP = self.args.FP) ,np.array([1]).reshape(1,-1).repeat(n2,axis = 0)],axis=1)
        # print('max,min atom fea BEFORE',np.max(H1),np.min(H1))
        if self.args.virtual_aromatic_atom:
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
        if self.args.fingerprintEdge:
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
        if self.args.only_dis_adj2:
            agg_adj2 = dm_all
        elif self.args.dis_adj2_with_adj1:# have som troubles to fix
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
        adj_graph_2 = np.copy(agg_adj1)

        pocket = (m1,m2)
        if self.args.supernode:#not useful in practice
            super_node_H = torch.mean(H,dim = 0)*0.0
            H = torch.cat([H,super_node_H.unsqueeze(0)],dim = 0)

            adj_graph_1 = np.concatenate([adj_graph_1, np.zeros_like(adj_graph_1[0][np.newaxis,:])], 0)
            adj_graph_1 = np.concatenate([adj_graph_1, np.zeros_like(adj_graph_1[:,0].reshape(-1,1))], 1)
            adj_graph_2 = np.concatenate([adj_graph_2, np.ones_like(adj_graph_2[0][np.newaxis,:])], 0)
            adj_graph_2 = np.concatenate([adj_graph_2, np.ones_like(adj_graph_2[:,0].reshape(-1,1))], 1)

            agg_adj1 = torch.cat([agg_adj1, torch.zeros_like(agg_adj1[0].unsqueeze(0))], dim = 0)
            agg_adj1 = torch.cat([agg_adj1, torch.zeros_like(agg_adj1[:,0].reshape(-1,1))], dim = 1)
            agg_adj2 = torch.cat([agg_adj2, torch.ones_like(agg_adj2[0].unsqueeze(0))], dim = 0)
            agg_adj2 = torch.cat([agg_adj2, torch.ones_like(agg_adj2[:,0].reshape(-1,1))], dim = 1)
        # time_s = time.time()
        item_1 = mol2graph(pocket,H,self.args,adj = adj_graph_1,n1 = n1,n2 = n2,\
            dm = (d1,d2) )
        # print('item_time:',time.time()-time_s)
        g ,full_g= preprocess_item(item_1, self.args,file_path,adj_graph_1,noise=False,size = size)
        full_g.edata['adj2'] = torch.tensor(dm_all).view(-1,1).contiguous().float()
        full_g.edata['adj1'] = agg_adj1.view(-1,1).contiguous().float()
        # full_g.edata['adj1'] = 
        # print('item_g:',time.time()-time_s)
        #item, args,file_path,adj,term,noise=False
        valid = torch.zeros((n1+n2,))
        if self.args.pred_mode == 'ligand':
            valid[:n1] = 1
        elif self.args.pred_mode == 'protein':
            valid[n1:] = 1
        elif self.args.pred_mode == 'supernode':
            valid = torch.zeros((n1+n2+1,))
            assert self.args.supernode == True,'plz setup your supernode in graph!'
            # assert len() == n1+n2+1 ,'no super node added to pocket graph check the args plz'
            valid[-1] = 1
        else:
            raise ValueError(f'not support this mode : {self.args.pred_mode} plz check the args')
        if self.args.loss_fn == 'mse_loss':
            Y = -float(key.split('-')[1].split('_')[0])
        else:
            if '_active' in key.split('/')[-1]:
                Y = 1 
                if self.args.add_logk_reg:
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
        # full_g.edata['adj2'] = agg_adj2.view(-1,1).contiguous()
        return g,full_g,Y


    
if __name__ == "__main__":

    import argparse
    from rdkit import RDLogger
    from dgl import save_graphs, load_graphs
    import dgl
    RDLogger.DisableLog('rdApp.*')
    # from train import get_args_from_json
    from torch.utils.data import DataLoader
    from prefetch_generator import BackgroundGenerator
    class DataLoaderX(DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())     
    # from train
    parser = argparse.ArgumentParser(description='json param')
    parser.add_argument("--json_path", help="file path of param", type=str, \
        default='/home/caoduanhua/score_function/GNN/GNN_graphformer_pyg/train_keys/config_files/gnn_edge_3d_pos_dgl.json')

    # label_smoothing# temp_args = parser.parse_args()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(args_dict['json_path'], args_dict)
    args = argparse.Namespace(**args)
    # print (args)
    train_keys = glob.glob('/home/caoduanhua/score_function/data/general_refineset/refineset_active_pocket_without_h/*')
    save_dir = '/home/caoduanhua/score_function/data/dgl_full_graph/'
    
    pbar = tqdm(train_keys)
    for i,key in enumerate(pbar):
        try:
            with open(key, 'rb') as f:
                m1,m2= pickle.load(f)
        except:
            print('file: {} is not a valid file！'.format(key))
        n1,d1,adj1 = utils.get_mol_info(m1)
        n2,d2,adj2 = utils.get_mol_info(m2)
        # pocket = Chem.CombineMols(m1,m2)
        H1 = get_atom_graphformer_feature(m1,FP = args.FP)
        H2 = get_atom_graphformer_feature(m2,FP = args.FP)
        if args.virtual_aromatic_atom:
            adj1,H1,d1,n1 = utils.add_atom_to_mol(m1,adj1,H1,d1,n1)
            # print( adj1,H1,d1,n1)
            adj2,H2,d2,n2 = utils.add_atom_to_mol(m2,adj2,H2,d2,n2)
            # print( adj2,H2,d2,n2)
        # H = torch.from_numpy(np.concatenate([H1, H2], 0))
 
        dm = distance_matrix(d1,d2)
        dm_all = distance_matrix(np.concatenate([d1,d2],axis=0),np.concatenate([d1,d2],axis=0))

        full_g = dgl.from_networkx(nx.complete_graph(n1 + n2))#g.number_of_nodes()
        full_g = full_g.add_self_loop() 
        all_rel_pos_3d_with_noise = torch.from_numpy(pandas_bins(dm_all,num_bins = None,noise = False)).long()
        full_g.edata['all_rel_pos_3d'] = all_rel_pos_3d_with_noise.view(-1,1).contiguous()
        with open(os.path.join(save_dir,key.split('/')[-1]),'wb') as f:
            pickle.dump(full_g,f)
            f.close()
        
        if i %100 == 0:
            pbar.update(100)
            pbar.set_description("Processing %d remain %d "%(i,len(pbar)- i))


        

       