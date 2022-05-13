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
        if self.args.add_logk_reg:
            with open('/home/caoduanhua/score_function/data/general_refineset/datapro/logk_match.pkl','rb') as f:
                self.logk_match =  pickle.load(f)
            # self.logk_match = 
    def __len__(self):
        if self.debug:
            return 128
        return len(self.keys)
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        g,full_g,Y = map(list, zip(*samples))
        # print(Y)
        Y = torch.tensor(Y).long()

        batch_g = dgl.batch(g)
        batch_full_g = dgl.batch(full_g)
        return batch_g,batch_full_g, Y
    def __getitem__(self, idx):
        #idx = 0
        key = self.keys[idx]
        file_path = self.args.path_data_dir+'/'+key
        try:
            try:
                with open(key, 'rb') as f:
        
                    m1,m2= pickle.load(f)
            except:
                with open(key, 'rb') as f:
        
                    m1,_,m2,_= pickle.load(f)
        except:
            print('file: {} is not a valid file！'.format(key))
            return None
        n1,d1,adj1 = utils.get_mol_info(m1)
        n2,d2,adj2 = utils.get_mol_info(m2)
        # pocket = Chem.CombineMols(m1,m2)
        H1 = get_atom_graphformer_feature(m1,FP = self.args.FP)
        H2 = get_atom_graphformer_feature(m2,FP = self.args.FP)
        if self.args.virtual_aromatic_atom:
            adj1,H1,d1,n1 = utils.add_atom_to_mol(m1,adj1,H1,d1,n1)
            # print( adj1,H1,d1,n1)
            adj2,H2,d2,n2 = utils.add_atom_to_mol(m2,adj2,H2,d2,n2)
            # print( adj2,H2,d2,n2)
        H = torch.from_numpy(np.concatenate([H1, H2], 0))
        agg_adj1 = np.zeros((n1+n2, n1+n2))
        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2
        agg_adj2 = np.copy(agg_adj1)
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
        
        item_1 = mol2graph(pocket,H,self.args,adj = adj_graph_1,n1 = n1,n2 = n2,\
            dm = (d1,d2) )
        g,full_g = preprocess_item(item_1, self.args,file_path,adj_graph_1,noise=False,size = size)
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
        g.ndata['V'] = valid.long().reshape(-1,1)
        full_g.edata['adj2'] = agg_adj2.view(-1,1).contiguous()
        return g,full_g,Y

if __name__ == "__main__":

    import argparse
    from rdkit import RDLogger
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
    with open (args.train_keys, 'rb') as fp:
        train_keys = pickle.load(fp)
    train_keys,val_keys = random_split(train_keys, split_ratio=0.9, seed=0, shuffle=True)
    val_dataset = graphformerDataset(val_keys,args, args.data_path,args.debug)
    # print(val_dataset)
    val_dataloader = DataLoaderX(val_dataset, args.batch_size, \
        shuffle=False, num_workers = args.num_workers, collate_fn=val_dataset.collate,pin_memory=True)
    for g,full_g,Y in val_dataloader:
        try:
            print(g.num_nodes(),g.num_edges())
            print(full_g.num_nodes(),full_g.num_edges())
            # print(Y)/
        except :
            print('something error!')
    # data_dir = '../../data/pocket_data/'
    # save_dir = '../../data/pocket_data_path/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # keys = os.listdir(data_dir)
    # pbar = tqdm(keys)
    # for i,key in enumerate(pbar):
    #     if not os.path.exists(save_dir + key):

    #         with open(data_dir+key, 'rb') as f:
    #             m1,_,m2,_ = pickle.load(f)

            
    #         n1 = m1.GetNumAtoms()
    #         c1 = m1.GetConformers()[0]
    #         d1 = np.array(c1.GetPositions())
    #         adj1 = GetAdjacencyMatrix(m1)+np.eye(n1)
            
    #         n2 = m2.GetNumAtoms()
    #         c2 = m2.GetConformers()[0]
    #         d2 = np.array(c2.GetPositions())
    #         adj2 = GetAdjacencyMatrix(m2)+np.eye(n2)
            
    #         agg_adj1 = np.zeros((n1+n2, n1+n2))
    #         agg_adj1[:n1, :n1] = adj1
    #         agg_adj1[n1:, n1:] = adj2
    #         agg_adj2 = np.copy(agg_adj1)
    #         dm = distance_matrix(d1,d2)
    #         dm = np.where(dm <5.0 ,1,0)

    #         agg_adj2[:n1,n1:] = np.copy(dm)
    #         agg_adj2[n1:,:n1] = np.copy(np.transpose(dm))
            
    #         shortest_path_result_mol, path_mol = algos.floyd_warshall(agg_adj1)
    #         shortest_path_result_pro, path_pro = algos.floyd_warshall(agg_adj2)
    #         with open(save_dir+key ,'ab') as f:
    #             pickle.dump((shortest_path_result_mol, path_mol,shortest_path_result_pro, path_pro),f)
    #             f.close()
        
    #     if i %10 == 0:

    #         pbar.update(10)
    #         pbar.set_description("Processing %d remain %d "%(i,len(pbar)- i))


        

       