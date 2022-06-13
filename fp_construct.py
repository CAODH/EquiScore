import pickle
import rdkit
import prolif as plf
import numpy as np
import pandas as pd
from collections import defaultdict
#将分子中的原子都标记上原子编号
def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol
def getNonBondPair(m1,m2):
    # 计算指纹 ，
    fp = plf.Fingerprint()
    prot = plf.Molecule(m2)
    ligand = plf.Molecule.from_rdkit(m1)
    fp.run_from_iterable([ligand],prot,progress=False,n_jobs = 1)
    df = fp.to_dataframe(return_atoms=True)
    # 找到非键相互作用的原子对的idx ,和rdkit的idx对应起来 构建残基名字+ 对应编号和rdkit id 的map
    res_to_idx = defaultdict(dict)
    for atom in m2.GetAtoms():
        # prot.residues
        atom_idx = atom.GetIdx()
        res_name = str(plf.residue.ResidueId.from_atom(atom))
        res_to_idx[res_name][len(res_to_idx[res_name])] = atom_idx
    # 与rdkit id 对应，用于后续加入任务
    nonbond_pairs = []
    inter_types = []
    for key in df.keys():
        ligand_name,res_name,inter_type = key
        lig_atom_num,res_atom_num = df[key][0]
        pdb_atom_idx = res_to_idx[res_name][res_atom_num]
        nonbond_pairs.append((lig_atom_num,pdb_atom_idx))  
        inter_types.append(inter_type) 

    return nonbond_pairs,inter_types