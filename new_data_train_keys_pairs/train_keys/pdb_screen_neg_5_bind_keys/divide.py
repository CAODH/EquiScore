import pickle
import prolif as plf
import numpy as np
import pandas as pd
import glob
import torch 
import sys
import shutil
import os
# from torch.nn.functional impor
from collections import defaultdict,Counter,OrderedDict
import matplotlib.pyplot as plt
import random
# random.seed(seed)

# utils
from typing import List,Dict,Tuple
# define a func to clip sample numbers in single uniport
def SamplePairs(pairs:List,threshold:int=400,ratio:float = 0.15,seed:int = 42)-> List:
    # print(pairs)
    try:
        import random
        random.seed(seed)
        if len(pairs) <= threshold:
            return pairs
        else:
            return random.sample(pairs,max(int(len(pairs)*ratio),threshold))    
    except TypeError:
        return 'TypeError'
def SelectTrainValUniport(eight_class_remove_duplicated,rate = 0.12,seed = 42):
    np.random.seed(seed)
    val_uniport_ids = []
    all_uniport_ids = []
    # val_uniport_pdb_ids =[]
    for uniport_set in eight_class_remove_duplicated:
        uniport_set = list(uniport_set)
        all_uniport_ids.extend(uniport_set)

        size = int(rate*len(uniport_set))
        val_uniport_ids.extend(np.random.choice(uniport_set,size = size))
    # for uniport_id in val_uniport_ids:
    train_uniport_ids = list(set(all_uniport_ids) - set(val_uniport_ids))
        # val_uniport_pdb_ids.extend(uniport_to_pdb_dict[uniport_id])
    return train_uniport_ids,val_uniport_ids
def SwapTuple(pair:Tuple)->List:
    new_pair = []
    new_pair.append(pair[1])
    new_pair.append(pair[0])
    new_pair.append(0)
    return new_pair
# define a reverse label func
def ReversePartLabel(pairs:List,ratio:float=0.5,seed:int=42)->List:
    ''' 
    reverse user-specified ratio pair to reverse label !
    input: pairs ,(active ,decoy,1) or (active1,active2,1)
    return new_pairs,reverse part pair label

    '''
    random.seed(seed)
    k = int(ratio*len(pairs))
    
    sampled_pairs = random.sample(pairs,k)
    rest_pairs = list(set(pairs) - set(sampled_pairs))
    rest_pairs = [list(_) + [1] for _ in rest_pairs]
    # reverse sampled pairs label and the order of the item!
    sampled_pairs = list(map(SwapTuple,sampled_pairs))
    # print(rest_pairs,sampled_pairs)
    rest_pairs = sampled_pairs + rest_pairs 
    return rest_pairs
def AddPathToPair(pairs,path_name = '/home/caoduanhua/score_function/data/D-PDBbind_PDBscreen/PDB_bind_active_pocket/',flag='_ligand_active_0'):
    return [(os.path.join(path_name,i  + flag),os.path.join(path_name,j  + flag)) for i,j in pairs]

def GetScreenDecoyPairs(data_dir,names,fast_num = 5,active_names = None):
    # data_name = []
    pro_decoy_pro = defaultdict(list)
    for i in names:
        pro = i.split('-')[0]
        if active_names is None:
            pro_decoy_pro[pro].append(i)
        else:
            if pro in active_names:
                pro_decoy_pro[pro].append(i)

    pro_decoy_pro_5 = defaultdict(list)
    pairs = []
    active_path = '/home/caoduanhua/score_function/data/D-PDBbind_PDBscreen/PDB_screen_active_-5_pocket/'
    # print(' pro_decoy_pro_5',len( pro_decoy_pro_5))
    for key in pro_decoy_pro.keys():
        if len(pro_decoy_pro[key]) <= fast_num:
            # print(key)

            pro_decoy_pro_5[key] = pro_decoy_pro[key]
        else:
            pro_decoy_pro_5[key] =pro_decoy_pro[key][:fast_num]

        pairs.extend([(os.path.join(active_path,key + '_active_0'),os.path.join(data_dir,decoy)) for decoy in pro_decoy_pro_5[key]])
    pro_decoy_pro_5 = sum(pro_decoy_pro_5.values(),[])
    print('mean of actives : decoys in cross_decoys',np.mean(list(dict(Counter([i.split('_')[2].split('-')[1] for i in pro_decoy_pro_5])).values())))
    # print('all decoy ',len(pro_decoy_pro_5)/len(active_names))
    return pairs
def GetPDBBindDecoyPairs(data_dir,names,fast_num = 5,active_names = None):
    
    # data_name = []
    pro_decoy_pro = defaultdict(list)
    for i in names:
        pro = i.split('_')[0]
        if active_names is None:
            pro_decoy_pro[pro].append(i)
        else:
            if pro in active_names:
                pro_decoy_pro[pro].append(i)

    pro_decoy_pro_5 = defaultdict(list)
    pairs = []
    active_path = '/home/caoduanhua/score_function/data/D-PDBbind_PDBscreen/PDB_bind_active_pocket/'
    for key in pro_decoy_pro.keys():
        if len(pro_decoy_pro[key]) <= fast_num:
            # print(key)

            pro_decoy_pro_5[key] = pro_decoy_pro[key]
        else:
            pro_decoy_pro_5[key] =pro_decoy_pro[key][:fast_num]
        pairs.extend([(os.path.join(active_path,key + '_ligand_active_0'),os.path.join(data_dir,decoy)) for decoy in pro_decoy_pro_5[key]])
    # active_path = '/home/caoduanhua/score_function/data/D-PDBbind_PDBscreen/PDB_bind_active_pocket/'

    pro_decoy_pro_5 = sum(pro_decoy_pro_5.values(),[])
    print('mean of actives : decoys in cross_decoys',np.mean(list(dict(Counter([i.split('_')[2] for i in pro_decoy_pro_5])).values())))
    return pairs

############################################ get pdbbind pairs############################################
# step1
# load all uniport id map pairs in bdbbind
with open('/home/caoduanhua/score_function/data/uniport_analysis/pair_sample/all_uniport_map_pairs.pkl','rb') as f:
    all_uniport_map_pairs = pickle.load(f)
    f.close()
# load remove duplicated uniport id to stratifited sampleing uniport ids as train/val
with open('/home/caoduanhua/score_function/data/uniport_analysis/pair_sample/selected_eight_class_uniport_remove_duplicated','rb') as f:
    remove_duplicated_uniport_sets = pickle.load(f)
    f.close()
# load  duplicated uniport id to stratifited sampleing uniport ids as test
with open('/home/caoduanhua/score_function/data/uniport_analysis/pair_sample/selected_eight_class_uniport_duplicated','rb') as f:
    duplicated_uniport_sets = pickle.load(f)
    f.close()
######################## set a random mode and a uniport mode#########################
# sum(len(i) for i in  duplicated_uniport_sets)
train_uniport,val_uniport = SelectTrainValUniport(remove_duplicated_uniport_sets,rate = 0.12,seed = 42)
test_uniport = sum([list(i) for i in duplicated_uniport_sets],[])
print('train uniport id ratio in all remove duplicated uniport ids: ',len(train_uniport)/(len(val_uniport) + len(train_uniport)))
# sample selected pairs
# step 1
train_pairs = []
for uniport in train_uniport:
    train_pairs.extend(AddPathToPair(SamplePairs(all_uniport_map_pairs[uniport])))
val_pairs = []
for uniport in val_uniport:
    val_pairs.extend(AddPathToPair(SamplePairs(all_uniport_map_pairs[uniport])))   
test_pairs = []
for uniport in test_uniport:
    test_pairs.extend(AddPathToPair(SamplePairs(all_uniport_map_pairs[uniport])))   
print('train pair ratio in all remove duplicated pairs',len(train_pairs)/(len(val_pairs) + len(train_pairs)))
print('test pair ratio in all train + val pairs : {} , \nnumber of test pairs : {}'.\
    format(len(test_pairs)/(len(val_pairs) + len(train_pairs)),len(test_pairs)))
print('active pair done ,lets procress the active-decoy pairs!')

# ########################### load duplicated pdb ids to filter dataset ###########################
with open('/home/caoduanhua/score_function/data/uniport_analysis/duplicated_with_independent_pdb_ids_from_uniport_id/all_dulicatedpdb_ids.pkl','rb') as f:
    duplicated_targets = pickle.load(f)
    print('duplicated tragets: ',len(duplicated_targets))

############################### pdb active decoy pairs ############################################
#
# valid_keys = glob.glob('/home/caoduanhua/score_function/data/pdb_screen/active_pocket/*')
valid_keys =glob.glob('/home/caoduanhua/score_function/data/D-PDBbind_PDBscreen/PDB_bind_active_pocket/*')
# valid_keys +=glob.glob('/home/caoduanhua/score_function/data/general_refineset/generalset_active_pocket_without_h/*')
active_pros = set([v.split('/')[-1].split('_')[0] for v in valid_keys])
# dude_gene =  set(OrderedDict.fromkeys([v.split('/')[-1].split('_')[0] for v in valid_keys]))
print('len of the before remove duplicated target: ',len(active_pros))
# dude_gene = dude_gene - duplicated_targets
active_pros_duplicated = [pdb for pdb in active_pros if pdb in duplicated_targets]
active_pros = [pdb for pdb in active_pros if pdb not in duplicated_targets]

print('len of the after remove duplicated target: ',len(active_pros))
print('remove done!')
print('pdb actives pros :',len(active_pros))
cross_decoys= os.listdir('/home/caoduanhua/score_function/data/D-PDBbind_PDBscreen/PDB_bind_cross_decoy_pocket/')
cross_decoys_pros = set([v.split('/')[-1].split('_')[0] for v in cross_decoys])
print('cross_decoys_pros :',len(cross_decoys_pros))
cross_decoys_dir = '/home/caoduanhua/score_function/data/D-PDBbind_PDBscreen/PDB_bind_cross_decoy_pocket/'
pdbDecoysPairs = GetPDBBindDecoyPairs(cross_decoys_dir,cross_decoys,fast_num=5,active_names = active_pros)
pdbDecoysPairs_duplicated = GetPDBBindDecoyPairs(cross_decoys_dir,cross_decoys,fast_num=5,active_names = active_pros_duplicated)
print(' pdbbind len of pairs: ',len(pdbDecoysPairs))
print(' pdbbind len of pairs: ',len(pdbDecoysPairs_duplicated))
############################### screen active decoy pairs ############################################
import pickle
with open('/home/caoduanhua/score_function/data/pdb_screen/score_bins_name_file/pdbscreen_active_5_2.pkl','rb') as f:
    data_names = pickle.load(f)
data_names = [i.split('.')[0].replace('_active','') for i in data_names]

valid_keys_screen = glob.glob('/home/caoduanhua/score_function/data/D-PDBbind_PDBscreen/PDB_screen_active_-5_pocket/*')
# filter by score -5
valid_keys_screen = [ i for i in valid_keys_screen if i.split('/')[-1].replace('_active_0','') in data_names]
active_pros_screen = set([v.split('/')[-1].replace('_active_0','') for v in valid_keys_screen])

active_pros_screen_duplicated = [screen for screen in active_pros_screen if screen.split('_')[0] in duplicated_targets]
print('len of the duplicated target: ',len(active_pros_screen_duplicated))
active_pros_screen = [screen for screen in active_pros_screen if screen.split('_')[0] not in duplicated_targets]
# dude_gene =  set(OrderedDict.fromkeys([v.split('/')[-1].split('_')[0] for v in valid_keys]))
print('len of the after remove duplicated target: ',len(active_pros_screen))

# print('pdb screen actives pros :',len(active_pros_screen))
cross_decoys_screen= os.listdir('/home/caoduanhua/score_function/data/D-PDBbind_PDBscreen/PDB_screen_cross_decoy_pocket/')
cross_decoys_pros_screen = set([v.split('-')[0] for v in cross_decoys_screen])
print('cross_decoys_pros :',len(cross_decoys_pros_screen))
cross_decoys_dir_screen = '/home/caoduanhua/score_function/data/D-PDBbind_PDBscreen/PDB_screen_cross_decoy_pocket/'
screenDecoyPairs= GetScreenDecoyPairs(cross_decoys_dir_screen,cross_decoys_screen,fast_num=6,active_names = active_pros_screen) 
screenDecoyPairs_duplicated = GetScreenDecoyPairs(cross_decoys_dir_screen,cross_decoys_screen,fast_num=6,active_names = active_pros_screen_duplicated)
print('  screen len of pairs: ',len(screenDecoyPairs))
print('  screen len of duplicated pairs: ',len(screenDecoyPairs_duplicated))

###################################### get all pairs done  ,then save as train val and test keys #######################################
test_pairs = test_pairs + screenDecoyPairs_duplicated  + screenDecoyPairs_duplicated
train_pairs = train_pairs + val_pairs + screenDecoyPairs + pdbDecoysPairs
print('reverse label default ratio = 0.5')
test_pairs = ReversePartLabel(test_pairs)
print('positive ratio in reversa label test pairs: ',np.sum([i[-1] for i in test_pairs])/len(test_pairs))
train_pairs = ReversePartLabel(train_pairs)
print('positive ratio in reversa label train pairs: ',np.sum([i[-1] for i in train_pairs])/len(train_pairs))
print ('Num train keys: ', len(train_pairs))
# print ('Num val keys: ', len(val_keys))
print ('Num test keys: ', len(test_pairs))
# print('train/val = ',len(train_keys)/len(val_keys))
# print('NUm uniport ids',len(val_uniport_ids))
with open('train_keys.pkl', 'wb') as f:
    pickle.dump(train_pairs, f)
with open('test_keys.pkl', 'wb') as f:
    pickle.dump(test_pairs, f)
# with open('val_keys.pkl', 'wb') as f:
#     pickle.dump(val_keys, f)