from graphformer_utils import *

from  dataset import *
import os
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
    def __len__(self):
        if self.debug:
            return 128
        return len(self.keys)

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
        if self.args.fundation_model == 'graphformer':

            H1 = get_atom_graphformer_feature(m1,FP = self.args.FP)
            H2 = get_atom_graphformer_feature(m2,FP = self.args.FP)
        if self.args.fundation_model == 'paper':

            H1 = get_atom_feature(m1, True,FP = self.args.FP)
            H2 = get_atom_feature(m2, False,FP = self.args.FP)
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
        if self.args.only_dis_adj2:
            agg_adj2 = distance_matrix(np.concatenate([d1,d2],axis=0),np.concatenate([d1,d2],axis=0))
        elif self.args.dis_adj2_with_adj1:# have som troubles to fix
            dm_all = distance_matrix(np.concatenate([d1,d2],axis=0),np.concatenate([d1,d2],axis=0))
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
        if self.args.dis_adj2_with_adj1:
            dm_all = np.where(dm_all <5.0 ,1,0)
            adj_graph_2  = np.where((dm_all+adj_graph_2) < 1 ,0,1)
        else:
            dm = np.where(dm <5.0 ,1,0)
            adj_graph_2[:n1,n1:] = np.copy(dm)
            adj_graph_2[n1:,:n1] = np.copy(np.transpose(dm))
        pocket = Chem.CombineMols(m1,m2)
        # if self.args.fundation_model == 'graphformer':
        #     H = torch.from_numpy(get_atom_graphformer_feature(pocket))
        # if self.args.fundation_model == 'paper':
        #     H1 = get_atom_feature(m1, True)
        #     H2 = get_atom_feature(m2, False)
        #     H = torch.from_numpy(np.concatenate([H1, H2], 0))
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
        item_1 = mol2graph(pocket,H,self.args,adj = adj_graph_1,n1 = n1,n2 = n2,dm = dm)
        item_1 = preprocess_item(item_1, self.args,file_path,adj_graph_1,noise=False,size = size)#item, args,file_path,adj,term,noise=False
        #
        item_2 = mol2graph(pocket,H,self.args,adj = adj_graph_2,n1 = n1,n2 = n2,dm = None)
        item_2 = preprocess_item(item_2, self.args,file_path,adj_graph_2,term = 'term_2',noise=False,size = None)

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
            Y = 1 if '_active' in key else 0
        # print(key)
        
        sample = {
                'H':item_2['x'], \
                'A1': agg_adj1, \
                'A2': agg_adj2, \
                'key': size, \
                'attn_bias':item_2['attn_bias'],\
                'attn_edge_type_1':item_1['attn_edge_type'],\
                'attn_edge_type_2':item_2['attn_edge_type'],\
                'rel_pos_1':item_1['rel_pos'],\
                'rel_pos_2':item_2['rel_pos'],\
                'in_degree_1':item_1['in_degree'],\
                'in_degree_2':item_2['in_degree'],\
                'out_degree_1':item_1['out_degree'],\
                'out_degree_2':item_2['out_degree'],\
                'edge_input_1':item_1['edge_input'],\
                'edge_input_2':item_2['edge_input'],\
                'all_rel_pos_3d':item_1['all_rel_pos_3d'],\
                'V': valid, \
                'Y': Y}
        # print(item_2['x'])

        return sample


class Batch():
    def __init__(self,H=None, \
            A1=None, \
            A2=None, \
            attn_bias=None,\
            attn_edge_type_1=None,\
            attn_edge_type_2=None,\
            rel_pos_1=None,\
            rel_pos_2=None,\
            in_degree_1=None,\
            in_degree_2=None,\
            out_degree_1=None,\
            out_degree_2=None,\
            edge_input_1=None,\
            edge_input_2=None,\
            all_rel_pos_3d=None,\
            V =None,key=None,Y=None):
        super(Batch, self).__init__()


    # def set_value(self):

        self.H = H
        
        self.in_degree_1, self.out_degree_1,self.in_degree_2, self.out_degree_2= in_degree_1, out_degree_1,in_degree_2, out_degree_2
        self.A1, self.A2 = A1,A2
        self.attn_bias, self.attn_edge_type_1, self.rel_pos_1 = attn_bias, attn_edge_type_1, rel_pos_1
        self.attn_edge_type_2, self.rel_pos_2 = attn_edge_type_2, rel_pos_2
        self.edge_input_1,self.edge_input_2 = edge_input_1,edge_input_2
        self.all_rel_pos_3d = all_rel_pos_3d
        self.V,self.key,self.Y = V,key,Y
    def get_att(self):
        return self.H, \
            self.A1, \
            self.A2, \
            self.attn_bias,\
            self.attn_edge_type_1,\
            self.attn_edge_type_2,\
            self.rel_pos_1,\
            self.rel_pos_2,\
            self.in_degree_1,\
            self.in_degree_2,\
            self.out_degree_1,\
            self.out_degree_2,\
            self.edge_input_1,\
            self.edge_input_2,\
            self.all_rel_pos_3d,\
            self.V ,self.key,self.Y
    def get_from_list(self,*data_list):
        self.H, \
            self.A1, \
            self.A2, \
            self.attn_bias,\
            self.attn_edge_type_1,\
            self.attn_edge_type_2,\
            self.rel_pos_1,\
            self.rel_pos_2,\
            self.in_degree_1,\
            self.in_degree_2,\
            self.out_degree_1,\
            self.out_degree_2,\
            self.edge_input_1,\
            self.edge_input_2,\
            self.all_rel_pos_3d,\
            self.V ,self.key,self.Y = data_list

    def to(self, device):
        self.H =self.H.to(device) if type(self.H) is torch.Tensor else self.H
        self.in_degree_1 = self.in_degree_1.to(device) if type(self.in_degree_1) is torch.Tensor else self.in_degree_1
        self.out_degree_1 =  self.out_degree_1.to(device) if type(self.out_degree_1) is torch.Tensor else self.out_degree_1
        self.in_degree_2 = self.in_degree_2.to(device) if type(self.in_degree_2) is torch.Tensor else self.in_degree_2
        self.out_degree_2= self.out_degree_2.to(device) if type(self.out_degree_2) is torch.Tensor else self.out_degree_2
        self.A1 = self.A1.to(device) if type(self.A1) is torch.Tensor else self.A1
        self.A2 = self.A2.to(device) if type(self.A2) is torch.Tensor else self.A2
        self.attn_bias =  self.attn_bias.to(device) if type(self.attn_bias) is torch.Tensor else self.attn_bias
        self.attn_edge_type_1 = self.attn_edge_type_1.to(device) if type(self.attn_edge_type_1) is torch.Tensor else self.attn_edge_type_1
        self.rel_pos_1 = self.rel_pos_1.to(device) if type(self.rel_pos_1) is torch.Tensor else self.rel_pos_1
        self.attn_edge_type_2=self.attn_edge_type_2.to(device) if type(self.attn_edge_type_2) is torch.Tensor else self.attn_edge_type_2
        self.rel_pos_2 = self.rel_pos_2.to(device) if type(self.rel_pos_2) is torch.Tensor else self.rel_pos_2
        self.edge_input_1=self.edge_input_1.to(device) if type(self.edge_input_1) is torch.Tensor else self.edge_input_1
        self.edge_input_2 = self.edge_input_2.to(device) if type(self.edge_input_2) is torch.Tensor else self.edge_input_2
        self.all_rel_pos_3d =self.all_rel_pos_3d.to(device) if type(self.all_rel_pos_3d) is torch.Tensor else self.all_rel_pos_3d
        self.V=self.V.to(device) if type(self.V) is torch.Tensor else self.V
        self.key=self.key.to(device) if type(self.key) is torch.Tensor else self.key
        self.Y =self.Y.to(device) if type(self.Y) is torch.Tensor else self.Y
        return self

    def __len__(self):
        return self.H.size(0)
def collate_fn(batch):
    max_natoms = max([len(item['H']) for item in batch if item is not None])
    batch = [item for item in batch if item is not None]
    sample = batch[0]
    atom_dim = sample['H'].size(-1)
    
    H = torch.zeros((len(batch), max_natoms, atom_dim))
    Y = torch.zeros((len(batch),))
    V = torch.zeros((len(batch), max_natoms))
    attn_bias = torch.zeros((len(batch), max_natoms, max_natoms))
#找出那些是None
    if sample['key'] is None:
        keys = None
    else:
        keys = []
    if sample['attn_edge_type_1'] is None:
        attn_edge_type_1 = None
        attn_edge_type_2 = None
    else:
        edge_dim = sample['attn_edge_type_1'].size(-1)
        attn_edge_type_1 =torch.zeros((len(batch), max_natoms, max_natoms,edge_dim)).long()
        attn_edge_type_2 =torch.zeros((len(batch), max_natoms, max_natoms,edge_dim)).long()
    if sample['rel_pos_1'] is None:
        rel_pos_1 = None
        rel_pos_2 = None
    else:
        rel_pos_1 =torch.zeros((len(batch), max_natoms, max_natoms)).long()
        rel_pos_2 =torch.zeros((len(batch), max_natoms, max_natoms)).long()
    if sample['in_degree_1'] is None:
        in_degree_1 = None
        in_degree_2 = None
    else:
        in_degree_1 =torch.zeros((len(batch), max_natoms)).long()
        in_degree_2 =torch.zeros((len(batch), max_natoms)).long()
    if sample['out_degree_1'] is None:
        out_degree_1 = None
        out_degree_2 = None
    else:
        out_degree_1 =torch.zeros((len(batch), max_natoms)).long()
        out_degree_2 =torch.zeros((len(batch), max_natoms)).long()
    if sample['edge_input_1'] is None:
        edge_input_1 = None
        edge_input_2 = None
    else:
        max_dist =  max([item['edge_input_1'].size(-2) for item in batch if item is not None])
        dim_input = sample['edge_input_1'].size(-1)
        edge_input_1 =torch.zeros((len(batch), max_natoms,max_natoms,max_dist,dim_input)).long()
        edge_input_2 =torch.zeros((len(batch), max_natoms,max_natoms,max_dist,dim_input)).long()
    if sample['all_rel_pos_3d'] is None:
        all_rel_pos_3d = None
    else:
        all_rel_pos_3d = torch.zeros((len(batch), max_natoms, max_natoms)).long()
    if sample['A1'] is None:
        A1 = None
        A2 = None
    else:
        A1 = torch.zeros((len(batch), max_natoms, max_natoms))
        A2 = torch.zeros((len(batch), max_natoms, max_natoms))

   
    
    for i in range(len(batch)):
        ligand_atoms,pro_atoms = batch[i]['key']
        natom = len(batch[i]['H'])
        H[i,:natom] = batch[i]['H']
        Y[i] = batch[i]['Y']
        V[i,:natom] = batch[i]['V']
        attn_bias[i,:natom,:natom ] = batch[i]['attn_bias']
        if sample['key'] is not None:
            keys.append(batch[i]['key'])
        if sample['attn_edge_type_1'] is not None:
           
            # edge_dim = sample['attn_edge_type_1'].size(-1)
            attn_edge_type_1[i,:natom,:natom,:] = batch[i]['attn_edge_type_1'].long()#np.zeros((len(batch), max_natoms, max_natoms,edge_dim))
            attn_edge_type_2[i,:natom,:natom,:] = batch[i]['attn_edge_type_2'].long()
        if sample['rel_pos_1'] is not None:
            rel_pos_1[i,:ligand_atoms,:ligand_atoms] = batch[i]['rel_pos_1'].long()
            rel_pos_2[i,:ligand_atoms,:ligand_atoms] = batch[i]['rel_pos_1'].long()# =np.zeros((len(batch), max_natoms, max_natoms))
        if sample['in_degree_1'] is not None:
           
            in_degree_1[i,:natom] = batch[i]['in_degree_1'].long()#=np.zeros((len(batch), max_natoms))
            in_degree_2[i,:natom] = batch[i]['in_degree_1'].long()
        if sample['out_degree_1'] is not None:
            out_degree_1[i,:natom] = batch[i]['out_degree_1'].long()#=np.zeros((len(batch), max_natoms))
            out_degree_2[i,:natom] = batch[i]['out_degree_1'].long()
        if sample['edge_input_1'] is not None:

            dist =  batch[i]['edge_input_1'].size(-2)
            
            dim_input = sample['edge_input_1'].size(-1)
            edge_input_1[i,:natom,:natom,:dist,:] = batch[i]['edge_input_1'].long()# =np.zeros((len(batch), max_natoms,max_natoms,max_dist,dim_input))
            edge_input_2[i,:natom,:natom,:dist,:] = batch[i]['edge_input_2'].long()
        if sample['all_rel_pos_3d'] is not None:
            all_rel_pos_3d[i,:natom,:natom]=batch[i]['all_rel_pos_3d']# = np.zeros((len(batch), max_natoms, max_natoms))
        if sample['A1'] is not  None:
            A1[i,:natom,:natom]=batch[i]['A1']# = np.zeros((len(batch), max_natoms, max_natoms))
            A2[i,:natom,:natom]=batch[i]['A2']# = np.zeros((len(batch), max_natoms, max_natoms))

  
    return  Batch(H, \
            A1, \
            A2, \
            attn_bias,\
            attn_edge_type_1,\
            attn_edge_type_2,\
            rel_pos_1,\
            rel_pos_2,\
            in_degree_1,\
            in_degree_2,\
            out_degree_1,\
            out_degree_2,\
            edge_input_1,\
            edge_input_2,\
            all_rel_pos_3d,\
            V ,keys, Y)




if __name__ == "__main__":
    data_dir = '../../data/pocket_data/'
    save_dir = '../../data/pocket_data_path/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    keys = os.listdir(data_dir)
    pbar = tqdm(keys)
    for i,key in enumerate(pbar):
        if not os.path.exists(save_dir + key):

            with open(data_dir+key, 'rb') as f:
                m1,_,m2,_ = pickle.load(f)

            
            n1 = m1.GetNumAtoms()
            c1 = m1.GetConformers()[0]
            d1 = np.array(c1.GetPositions())
            adj1 = GetAdjacencyMatrix(m1)+np.eye(n1)
            
            n2 = m2.GetNumAtoms()
            c2 = m2.GetConformers()[0]
            d2 = np.array(c2.GetPositions())
            adj2 = GetAdjacencyMatrix(m2)+np.eye(n2)
            
            agg_adj1 = np.zeros((n1+n2, n1+n2))
            agg_adj1[:n1, :n1] = adj1
            agg_adj1[n1:, n1:] = adj2
            agg_adj2 = np.copy(agg_adj1)
            dm = distance_matrix(d1,d2)
            dm = np.where(dm <5.0 ,1,0)

            agg_adj2[:n1,n1:] = np.copy(dm)
            agg_adj2[n1:,:n1] = np.copy(np.transpose(dm))
            
            shortest_path_result_mol, path_mol = algos.floyd_warshall(agg_adj1)
            shortest_path_result_pro, path_pro = algos.floyd_warshall(agg_adj2)
            with open(save_dir+key ,'ab') as f:
                pickle.dump((shortest_path_result_mol, path_mol,shortest_path_result_pro, path_pro),f)
                f.close()
        
        if i %10 == 0:

            pbar.update(10)
            pbar.set_description("Processing %d remain %d "%(i,len(pbar)- i))


        

       