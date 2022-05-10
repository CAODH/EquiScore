'''1、 数据里面已经包含了ligand的mol 和pocket 的mol
处理数据需要分开两个分子处理一下，合并起来再处理一下，分别保存
2、分子特征暂时只用GNN_DTI里面提供的，后面再考虑grapformer里面的一些特征
3、拿到编码边的方法
4、拿到编码路径的方法（相对位置）
5、拿到编码空间距离的方法（GNN_DTI里面用的构建分子间的边就是这么用的）
6、拿到编码度的方法（one_hot）
以上编码方式都是可以训练的所以，需要在forward的时候做
写个脚本处理的数据都存到一个文件里面然后训练再读出来
'''
import torch
import numpy as np
# from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
from utils import *
from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
from dataset import *
import joblib
import numpy as np
import math
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix


import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos
import pickle


import os.path as osp
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
# ===================== BOND START =====================
possible_bond_type_list = list(range(32))
possible_bond_stereo_list = list(range(16))
possible_is_conjugated_list = [False, True]
possible_is_in_ring_list = [False, True]
possible_bond_dir_list = list(range(16))

def bond_to_feature_vector(bond):
    # 0
    bond_type = int(bond.GetBondType())
    assert bond_type in possible_bond_type_list

    bond_stereo = int(bond.GetStereo())
    assert bond_stereo in possible_bond_stereo_list

    is_conjugated = bond.GetIsConjugated()
    assert is_conjugated in possible_is_conjugated_list
    is_conjugated = possible_is_conjugated_list.index(is_conjugated)

    is_in_ring = bond.IsInRing()
    assert is_in_ring in possible_is_in_ring_list
    is_in_ring = possible_is_in_ring_list.index(is_in_ring)

    bond_dir = int(bond.GetBondDir())
    assert bond_dir in possible_bond_dir_list

    bond_feature = [
        bond_type,
        bond_stereo,
        is_conjugated,
        is_in_ring,
        bond_dir,
    ]
    return bond_feature

#by caodunahua under this line
def GetNum(x,allowed_set):
    try:
        return [allowed_set.index(x)]
    except:
        return [len(allowed_set) -1]

def atom_feature_graphformer(m, atom_i, i_donor, i_acceptor):

    atom = m.GetAtomWithIdx(atom_i)
    return np.array(GetNum(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    GetNum(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    GetNum(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    GetNum(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (10, 6, 5, 6, 1) --> total 28
#   ['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','other']) \
def atom_feature_attentive_FP(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=True):
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:
        results = GetNum(
          atom.GetSymbol(),
          ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H','other']) \
              + GetNum(atom.GetDegree(),[0, 1, 2, 3, 4, 5]) + \
                  [int(atom.GetFormalCharge()), int(atom.GetNumRadicalElectrons())] + \
                  GetNum(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2,'other'
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + GetNum(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + GetNum(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)
def get_atom_graphformer_feature(m,FP = False):
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        if FP:
            H.append(atom_feature_attentive_FP(m.GetAtomWithIdx(i),
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=True))
        else:
            H.append(atom_feature_graphformer(m, i, None, None))
    H = np.array(H)        
    # print('H in get func',H)
    return H      

# ===================== BOND END =====================
def convert_to_single_emb(x, offset=32):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

#===================3d position start ========================
def get_rel_pos(mol):
    try:
        new_mol = Chem.AddHs(mol)
        res = AllChem.EmbedMultipleConfs(new_mol, numConfs=10)
        ### MMFF generates multiple conformations
        res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        index = np.argmin([x[1] for x in res])
        energy = res[index][1]
        conf = new_mol.GetConformer(id=int(index))
    except:
        new_mol = mol
        AllChem.Compute2DCoords(new_mol)
        energy = 0
        conf = new_mol.GetConformer()

    atom_poses = []
    for i, atom in enumerate(new_mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return [[0.0, 0.0, 0.0]] * len(new_mol.GetAtoms())
        pos = conf.GetAtomPosition(i)
        atom_poses.append([pos.x, pos.y, pos.z])
    atom_poses = np.array(atom_poses, dtype=float)
    rel_pos_3d = cdist(atom_poses, atom_poses)
    return rel_pos_3d
#===================3d position end ========================
#===================data attris start ========================
# from dataset import *

def molEdge(mol,n1,n2,adj_mol = None):
    edges_list = []
    edge_features_list = []
    if len(mol.GetBonds()) > 0: # mol has bonds
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx() 
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
    # add virtual aromatic nodes feature
    # add mol 
    if adj_mol is None:
        return edges_list ,edge_features_list
    else:
        n = len(mol.GetAtoms())
        adj_mol -= np.eye(len(adj_mol))
        dm = adj_mol[n:n1,:n1]
        # adj_mol += np.eye(len(adj_mol))
        edge_pos = np.where(dm ==1)
        edges_list.extend([(i+ n,j) for (i,j) in zip(*edge_pos)])
        edge_features_list.extend([[33,17,3,3,17] for edge_tuple in zip(*edge_pos)])
        edges_list.extend([(j,i + n) for (i,j) in zip(*edge_pos)])
        edge_features_list.extend([[33,17,3,3,17] for edge_tuple in zip(*edge_pos)])
        adj_mol += np.eye(len(adj_mol))
    return edges_list ,edge_features_list
def pocketEdge(mol,n1,n2,adj_pocket = None):
    edges_list = []
    edge_features_list = []
    if len(mol.GetBonds()) > 0: # mol has bonds
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx() + n1
            j = bond.GetEndAtomIdx() + n1
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
    if adj_pocket is  None :
        return edges_list ,edge_features_list
    # add virtual aromatic nodes feature
    # add pocket
    else:
        n = len(mol.GetAtoms()) + n1
        all_n = len(adj_pocket)
        adj_pocket -= np.eye(all_n)
        dm = adj_pocket[n:,n1:]
        edge_pos = np.where(dm ==1)
        edges_list.extend([(i+ n,j + n1) for (i,j) in zip(*edge_pos)])
        edge_features_list.extend([[33,17,3,3,17] for edge_tuple in zip(*edge_pos)])
        edges_list.extend([(j + n1,i + n) for (i,j) in zip(*edge_pos)])
        edge_features_list.extend([[33,17,3,3,17] for edge_tuple in zip(*edge_pos)])
        adj_pocket += np.eye(all_n)
    return edges_list ,edge_features_list
def getEdge(mols,n1,n2,adj_in = None):
    num_bond_features = 5
    mol,pocket = mols
    mol1_edge_idxs,mol1_edge_attr = molEdge(mol,n1,n2,adj_mol = adj_in)
    mol2_edge_idxs,mol2_edge_attr = pocketEdge(pocket,n1,n2,adj_pocket = adj_in)
    edges_list = mol1_edge_idxs + mol2_edge_idxs
    edge_features_list = mol1_edge_attr + mol2_edge_attr

    if adj_in is  None:
        pass
    else:
        #加入虚拟边，然后为虚拟边加入特征向量 add fingerprint edges features
        dm = adj_in[:n1,n1:n2]
        edge_pos = np.where(dm == 1)
        edges_list.extend([(i,j+n1) for (i,j) in zip(*edge_pos)])
        edge_features_list.extend([[32,16,2,2,16] for edge_tuple in zip(*edge_pos)])
        edges_list.extend([(j+n1,i) for (i,j) in zip(*edge_pos)])
        edge_features_list.extend([[32,16,2,2,16] for edge_tuple in zip(*edge_pos)])
    if len(edges_list) == 0:
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)
    else:
        edge_index = np.array(edges_list, dtype = np.int64).T
        edge_attr = torch.tensor(edge_features_list, dtype = torch.int64)
    return edge_index,edge_attr

def mol2graph(mol,x,args,n1,n2,adj = None,dm = None):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    # x = get_atom_feature(mol,is_ligand=is_ligand)#array
    if args.edge_bias:
        edge_index, edge_attr= getEdge(mol,adj_in = adj,n1 = n1,n2 = n2)
    else:
        edge_index, edge_attr= None,None
    # attn

    if args.rel_3d_pos_bias and dm is not None:
        if len(dm) == 2:
            d1,d2 = dm
            rel_pos_3d = distance_matrix(np.concatenate([d1,d2],axis=0),np.concatenate([d1,d2],axis=0))
        else:
            rel_pos_3d = np.zeros((n1+n2, n1+n2))
            rel_pos_3d[:n1,n1:] = np.copy(dm)
            rel_pos_3d[n1:,:n1] = np.copy(np.transpose(dm))
    else:
        rel_pos_3d =  None
    graph = dict()

    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    # graph['num_nodes'] = len(x)
    graph['rel_pos_3d'] = rel_pos_3d
    return graph 
#=================== data attrs end ========================
#===================all data attrs process start ========================
import pandas as pd
import numpy as np
def pandas_bins(dis_matrix,num_bins = None,noise = False):
    if num_bins is None:
        num_bins = int((5-2.0)/0.05 + 1)
    if noise:
        t = np.random.laplace(0.001, 0.05)
        dis_matrix += t
    bins = [-1.0] + list(np.linspace(2.0,5,num_bins)) + [10000]
    shape = dis_matrix.shape
    bins_index = np.array(pd.cut(dis_matrix.flatten(),bins = bins,labels = [i for i in range(len(bins) -1)])).reshape(shape)
    return bins_index
def preprocess_item(item, args,file_path,adj,term='item_1',noise=False,size = None):
    edge_attr, edge_index, x  = item['edge_feat'], item['edge_index'], item['node_feat']
    N = x.size(0)
    # print('num features:',N)
    if args.fundation_model == 'graphformer':
        offset = 16 if args.FP else 10
        x = convert_to_single_emb(x,offset = offset)
    adj = torch.tensor(adj,dtype=torch.long)
    # edge feature here

    all_rel_pos_3d_with_noise = torch.from_numpy(pandas_bins(item['rel_pos_3d'],num_bins = None,noise = False)).long() \
        if args.rel_3d_pos_bias and  term == 'item_1' else None

    if args.rel_pos_bias and size:
        if os.path.exists(file_path):
            with open(file_path,'rb') as f:
                if term == 'term_2':
                    shortest_path_result, path,_,_ = pickle.load(f)
                else:
                    _,_,shortest_path_result, path = pickle.load(f)
        else:
            shortest_path_result, path = algos.floyd_warshall(adj[:size[0],:size[0]].numpy())
            # print('shortest path result ',)

        # max_dist = np.amax(shortest_path_result)
        rel_pos = torch.from_numpy((shortest_path_result)).long() #if args.rel_pos else None
        # print('shortest path result ',rel_pos.shape)

    else:
        rel_pos = None
    if args.edge_bias and term == 'item_1':
        if len(edge_attr.shape) == 1:
            edge_attr = edge_attr[:, None]
        # all_rel_pos_3d_with_noise = torch.from_numpy(algos.bin_rel_pos_3d_1(item['rel_pos_3d'], noise=noise)).long()
        # rel_pos_3d_attr = all_rel_pos_3d_with_noise[edge_index[0, :], edge_index[1, :]]
        # edge_attr = torch.cat([edge_attr, rel_pos_3d_attr[:, None]], dim=-1)
        attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
        attn_edge_type[edge_index[0, :], edge_index[1, :]] = convert_to_single_emb(edge_attr)# + 1#
        edge_input = None
    else:
        attn_edge_type = None
        edge_input = None
    # rel_pos = torch.from_numpy((shortest_path_result)).long() if args.rel_pos else None
    # attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float) # with graph token
    attn_bias = torch.zeros([N, N], dtype=torch.float) 
    assert len(attn_bias.shape ) == 2 ,print('attn_bias:',attn_bias) 
    # combine
    item['x'] = x
    # item['adj'] = adj
    item['attn_bias'] = attn_bias
    item['attn_edge_type'] = attn_edge_type#每条边的特征
    item['rel_pos'] = rel_pos
    adj_in,adj_out = adj.long().sum(dim=1).view(-1),adj.long().sum(dim=0).view(-1)

    item['in_degree'] = torch.where(adj_in > 8,9,adj_in) if args.in_degree_bias else None #每个结点的输入边的特征
    item['out_degree'] = torch.where(adj_out > 8,9,adj_out) if args.out_degree_bias else None
    item['edge_input'] = torch.from_numpy(edge_input).long() if edge_input is not None else None
    item['all_rel_pos_3d'] = all_rel_pos_3d_with_noise#torch.long
    #item['all_rel_pos_3d'] = torch.from_numpy(all_rel_pos_3d_with_noise).float() if all_rel_pos_3d_with_noise is not None else None
    return item



