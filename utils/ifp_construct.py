
import prolif as plf
from collections import defaultdict

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol
def get_nonBond_pair(m1,m2):
    """
    docstring: 
        calculate IFP(interaction fingerprint)
    input:
        m1: rdkit.Chem.Mol(ligand)
        m2: rdkit.Chem.Mol(protein)
    output:
        nonbond_pairs: list of tuple (ligand_atom_idx,protein_atom_idx)
        inter_types: list of interaction type
    """
    fp = plf.Fingerprint()
    prot = plf.Molecule(m2)
    ligand = plf.Molecule.from_rdkit(m1)
    fp.run_from_iterable([ligand],prot,progress=False,n_jobs = 1)
    df = fp.to_dataframe(return_atoms=True)
    # find IFP  atom pairs , match with rdkit id 
    res_to_idx = defaultdict(dict)
    for atom in m2.GetAtoms():
        atom_idx = atom.GetIdx()
        res_name = str(plf.residue.ResidueId.from_atom(atom))
        res_to_idx[res_name][len(res_to_idx[res_name])] = atom_idx

    nonbond_pairs = []
    inter_types = []
    for key in df.keys():
        ligand_name,res_name,inter_type = key
        lig_atom_num,res_atom_num = df[key][0]
        pdb_atom_idx = res_to_idx[res_name][res_atom_num]
        nonbond_pairs.append((lig_atom_num,pdb_atom_idx))  
        inter_types.append(inter_type) 

    return nonbond_pairs,inter_types