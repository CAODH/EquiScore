import torch
import numpy as np
from utils import *
from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
import numpy as np
import rdkit
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
import dgl
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
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
def bond_to_feature_vector(bond):
    """
    input: rdkit.Chem.rdchem.Bond
    output: bond_feature (list)
    
    """
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

def GetNum(x,allowed_set):
    """
    input: 
        x (int)
        allowed_set (list)
    output:    
        [index] (list)
    
    """
    try:
        return [allowed_set.index(x)]
    except:
        return [len(allowed_set) -1]
def get_aromatic_rings(mol:rdkit.Chem.Mol) -> list:

    """
    input: 
        rdkit.Chem.Mol
    output:
       aromaticatoms rings (list)
    """
    aromaticity_atom_id_set = set()
    rings = []
    for atom in mol.GetAromaticAtoms():
        aromaticity_atom_id_set.add(atom.GetIdx())
    # get ring info 
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        ring_id_set = set(ring)
        # check atom in this ring is aromaticity
        if ring_id_set <= aromaticity_atom_id_set:
            rings.append(list(ring))
    return rings
def add_atom_to_mol(mol:rdkit.Chem.Mol,adj:np.array,H:np.array,d:np.array,n:int):
    """
    docstring: 
        add virtual aromatic atom feature/adj/3d_positions to raw data
    input:
        mol: rdkit.Chem.Mol
        adj: adj matrix
        H: node feature
        node d: 3d positions
        n: node nums 
    """
    assert len(adj) == len(H),'adj nums not equal to nodes'
    rings = get_aromatic_rings(mol)
    num_aromatic = len(rings)
    h,b = adj.shape
    all_zeros = np.zeros((num_aromatic+h,num_aromatic+b))
    #add all zeros vector to bottom and right
    all_zeros[:h,:b] = adj
    for i,ring in enumerate(rings):
        all_zeros[h+i,:][ring] = 1
        all_zeros[:,h+i][ring] = 1
        all_zeros[h+i,:][h+i] = 1
        d = np.concatenate([d,np.mean(d[ring],axis = 0,keepdims=True)],axis = 0)
        H  = np.concatenate([H,np.array([15]*(H.shape[1]))[np.newaxis]],axis = 0)
    assert len(all_zeros) == len(H),'adj nums not equal to nodes'
    return all_zeros,H,d,n+num_aromatic
def get_mol_info(m1):
    """
    input: 
        rdkit.Chem.Mol
    output:
        n1: node nums
        d1: 3d positions
        adj1: adj matrix
    """
    n1 = m1.GetNumAtoms()
    c1 = m1.GetConformers()[0]
    d1 = np.array(c1.GetPositions())
    adj1 = GetAdjacencyMatrix(m1)+np.eye(n1)
    return n1,d1,adj1
def atom_feature_graphformer(m, atom_i, i_donor, i_acceptor):
    """
    docstring:
        atom feature as same as graphformer
    input:
        m: rdkit.Chem.Mol
        atom_i: atom index
        i_donor: H donor or not 
        i_acceptor: H acceptor or not 
    output:
        atom feature (list)
    """
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
    """
    docstring:
        atom feature as same as attentiveFP
    input:
        m: rdkit.Chem.Mol
        atom_i: atom index
        i_donor: H donor or not 
        i_acceptor: H acceptor or not 
    output:
        atom feature (array)
    """
                  
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

    return H      

# ===================== BOND END =====================
def convert_to_single_emb(x, offset=35):
    """
    docstring:
        merge multiple embeddings into one embedding

    """
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
#===================data attributes start ========================
# from dataset import *
def molEdge(mol,n1,n2,adj_mol = None):
    """
    docstring:
        get edges and edge features of mol
    input:
        mol: rdkit.Chem.Mol
        n1: number of atoms in mol
        # n2: number of atoms in adj_mol
        adj_mol: adjacent matrix of mol
    output:
        edges_list: list of edges
        edge_features_list: list of edge features
    """
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
    if adj_mol is None:
        return edges_list ,edge_features_list
    else:
        n = len(mol.GetAtoms())
        adj_mol -= np.eye(len(adj_mol))
        dm = adj_mol[n:n1,:n1]
        edge_pos_u,edge_pos_v = np.where(dm == 1)
        
        u,v = list((edge_pos_u + n)) +  list( edge_pos_v),list( edge_pos_v) + list((edge_pos_u + n))
        edges_list.extend([*zip(u,v)])
        edge_features_list.extend([[33,17,3,3,17]]*len(u))

    return edges_list ,edge_features_list
def pocketEdge(mol,n1,n2,adj_pocket = None):
    """"
    docstring:
        get edges and edge features of pocket
    input:
        mol: rdkit.Chem.Mol
        n1: number of atoms in mol
        # n2: number of atoms in adj_pocket
        adj_pocket: adjacent matrix of pocket
    output:
        edges_list: list of edges
        edge_features_list: list of edge features 
    """
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
        
        u,v = list((edge_pos_u + n)) +  list( edge_pos_v + n1),list( edge_pos_v + n1) + list((edge_pos_u + n))

        edges_list.extend([*zip(u,v)])
        edge_features_list.extend([[33,17,3,3,17]]*len(u))
   
    return edges_list ,edge_features_list
def getEdge(mols,n1,n2,adj_in = None):
    """
    Docstring:
        merge molEdge and pocketEdge
    input:
        mols: list of rdkit.Chem.Mol
        n1: number of atoms in mol
        n2: number of atoms in pocket
        adj_in: adjacent matrix of mol and pocket
    output:
        edges_list: list of edges
        edge_features_list: list of edge features
    """
    num_bond_features = 5
    mol,pocket = mols
    mol1_edge_idxs,mol1_edge_attr = molEdge(mol,n1,n2,adj_mol = adj_in)
    mol2_edge_idxs,mol2_edge_attr = pocketEdge(pocket,n1,n2,adj_pocket = adj_in)
    edges_list = mol1_edge_idxs + mol2_edge_idxs
    edge_features_list = mol1_edge_attr + mol2_edge_attr
    # add self edge
    u,v = np.where(np.eye(n1+n2) == 1)
    edges_list.extend([*zip(u,v)])
    edge_features_list.extend([[34,17,4,4,18]]*len(u))
    if adj_in is  None:
        pass
    else:
        #add virtual edge , add fingerprint edges features
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
    return edge_index,edge_attr


def mol2graph(mol,x,args,n1,n2,adj = None,dm = None):
    """
    dcostring:
        Converts mol to graph Data object
    input: 
        mol: rdkit.Chem.Mol
        x: node features
        args: args
        n1: number of atoms in mol
        n2: number of atoms in pocket
        adj: adjacent matrix of mol and pocket
        dm: distance matrix of mol and pocket
    output: 
        graph object
    """
    if args.edge_bias:
        edge_index, edge_attr= getEdge(mol,adj_in = adj,n1 = n1,n2 = n2)
    else:
        edge_index, edge_attr= None,None

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
def preprocess_item(item, args,adj):

    edge_attr, edge_index, x  = item['edge_feat'], item['edge_index'], item['node_feat']
    N = x.size(0)
    if args.model == 'EquiScore':
        offset = 16 if args.FP else 10
        x = convert_to_single_emb(x,offset = offset)
    if x.min()< 0:
        print('convert feat',x.min())
    
    adj = torch.tensor(adj,dtype=torch.long)
    # edge feature here
    g = dgl.graph((edge_index[0, :], edge_index[1, :]),num_nodes=len(adj))
    if args.lap_pos_enc:
        g.ndata['lap_pos_enc'] = get_pos_lp_encoding(adj.numpy(),pos_enc_dim = args.pos_enc_dim)
    g.ndata['x']  = x
    adj_in = adj.long().sum(dim=1).view(-1)
    adj_in = torch.where(adj_in < 0,0,adj_in)
    g.ndata['in_degree'] = torch.where(adj_in > 8,9,adj_in) if args.in_degree_bias else None
    g.edata['edge_attr'] = convert_to_single_emb(edge_attr)   
    return g



