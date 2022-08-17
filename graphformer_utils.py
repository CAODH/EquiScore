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
import networkx as nx
import dgl
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
def convert_to_single_emb(x, offset=35):
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
        edge_pos_u,edge_pos_v = np.where(dm == 1)
        # print('mol virtual edge',edge_pos_u,edge_pos_v)
        
        u,v = list((edge_pos_u + n)) +  list( edge_pos_v),list( edge_pos_v) + list((edge_pos_u + n))
        # print('mol virtual edge',u,v)
        edges_list.extend([*zip(u,v)])
        edge_features_list.extend([[33,17,3,3,17]]*len(u))
        # adj_mol += np.eye(len(adj_mol))
    # assert np.max(edges_list) < len(adj_mol),'edge_index must be less than nodes! ' 
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
    # add self edge feature
    if adj_pocket is  None :
        return edges_list ,edge_features_list
    # add virtual aromatic nodes feature
    # add pocket
    else:
        n = len(mol.GetAtoms()) + n1
        all_n = len(adj_pocket)
        adj_pocket -= np.eye(all_n)
        dm = adj_pocket[n:,n1:]
        edge_pos_u,edge_pos_v = np.where(dm == 1)
        # print('pocket virtual edge',edge_pos_u,edge_pos_v)
        
        u,v = list((edge_pos_u + n)) +  list( edge_pos_v + n1),list( edge_pos_v + n1) + list((edge_pos_u + n))
        # print('pocket virtual edge',u,v)
        edges_list.extend([*zip(u,v)])
        edge_features_list.extend([[33,17,3,3,17]]*len(u))
   
        # adj_pocket += np.eye(all_n)
    # assert np.max(edges_list) < len(adj_pocket),'edge_index must be less than nodes! ' 
    return edges_list ,edge_features_list
def getEdge(mols,n1,n2,adj_in = None):
    num_bond_features = 5
    mol,pocket = mols
    mol1_edge_idxs,mol1_edge_attr = molEdge(mol,n1,n2,adj_mol = adj_in)
    mol2_edge_idxs,mol2_edge_attr = pocketEdge(pocket,n1,n2,adj_pocket = adj_in)
    edges_list = mol1_edge_idxs + mol2_edge_idxs
    edge_features_list = mol1_edge_attr + mol2_edge_attr
    # add self edge
    u,v = np.where(np.eye(n1+n2) == 1)
    # u_new = np.concatenate([u,v],axis = 0)
    # v_new = np.concatenate([v,u],axis = 0)
    edges_list.extend([*zip(u,v)])
    edge_features_list.extend([[34,17,4,4,18]]*len(u))

    if adj_in is  None:
        pass
    else:
        #加入虚拟边，然后为虚拟边加入特征向量 add fingerprint edges features
        dm = adj_in[:n1,n1:]
        edge_pos_u,edge_pos_v = np.where(dm == 1)
        
        u,v = list(edge_pos_u) +  list((n1+ edge_pos_v)),list((n1+ edge_pos_v)) + list(edge_pos_u)

        edges_list.extend([*zip(u,v)])
        edge_features_list.extend([[32,16,2,2,16]]*len(u))
    if len(edges_list) == 0:
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)
    else:
        edge_index = np.array(edges_list, dtype = np.int64).T
        edge_attr = torch.tensor(edge_features_list, dtype = torch.int64)
        # assert np.max(edge_index) < len(adj_in),'edge_index must be less than nodes! ' 
    return edge_index,edge_attr


def mol2graph(mol,x,args,n1,n2,adj = None,dm = None):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    # x = get_atom_feature(mol,is_ligand=is_ligand)#array
    if args.edge_bias:
        # time_s = time.time()
        edge_index, edge_attr= getEdge(mol,adj_in = adj,n1 = n1,n2 = n2)
        # print()
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
import scipy.sparse as sp
def get_pos_lp_encoding(adj,pos_enc_dim = 8):
    A = sp.coo_matrix(adj)
    N = sp.diags(adj.sum(axis = 1).clip(1) ** -0.5, dtype=float)
    L = sp.eye(len(adj)) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    return lap_pos_enc
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
def preprocess_item(item, args):
    # noise = False
    # time_s = time.time()
    edge_attr, edge_index, x  = item['edge_feat'], item['edge_index'], item['node_feat']
    # print('get edge from  molgraph: ',time.time()-time_s)
    N = x.size(0)
    # print('num features:',N)
    if args.fundation_model == 'graphformer':
        offset = 16 if args.FP else 10
        x = convert_to_single_emb(x,offset = offset)
    adj = torch.tensor(adj,dtype=torch.long)
    # edge feature here
    g = dgl.graph((edge_index[0, :], edge_index[1, :]),num_nodes=len(adj))
    # print('get dgl_graph: ',time.time()-time_s)
    # 这里不包含self loop
    # g.add
    if args.lap_pos_enc:

        g.ndata['lap_pos_enc'] = get_pos_lp_encoding(adj.numpy(),pos_enc_dim = args.pos_enc_dim)
 
    g.ndata['x']  = x
    adj_in = adj.long().sum(dim=1).view(-1)
    adj_in = torch.where(adj_in < 0,0,adj_in)
    g.ndata['in_degree'] = torch.where(adj_in > 8,9,adj_in) if args.in_degree_bias else None
    # print('max() min()',max(adj_in),min(adj_in))
    g.edata['edge_attr'] = convert_to_single_emb(edge_attr)

    src,dst = np.where(np.ones_like(adj)==1)
    full_g = dgl.graph((src,dst))

    if args.rel_3d_pos_bias:
        all_rel_pos_3d_with_noise = torch.from_numpy(pandas_bins(item['rel_pos_3d'],num_bins = None,noise = args.noise)).long() 
        full_g.edata['rel_pos_3d'] = all_rel_pos_3d_with_noise.view(-1,1).contiguous()#torch.long
    
    return g,full_g



