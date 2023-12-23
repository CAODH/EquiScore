"""
----------------------------------------------------Attention-------------------------------------------------------
                please repalce the relative path (../../training_data/PDBscreen) to the absolute path first, 
                such as : /home/user-name/Equiscore/data/training_data/PDBscreen
                this is the most important setup for this script!
----------------------------------------------------Attention-------------------------------------------------------
"""
import pickle
from collections import OrderedDict
import random
import glob
import numpy as np
import os
random.seed(0)
from collections import defaultdict,Counter
'''
you should download the PDBscreen dataset first and put it in the ./data/training_data folder
for better control the data , We further divided the collected data into high quality(HQ) parts, and low quality(LQ) parts
to control the data quality, We performed docking processing separately for each part
###########################################################################################################################
PDBscreen_HQ_active_augment_2a_pose_pocket  PDBscreen_HQ_generated_decoy_pocket         PDBscreen_LQ_cross_decoy_pocket
PDBscreen_HQ_active_pocket                  PDBscreen_LQ_active_augment_2a_pose_pocket  PDBscreen_LQ_generated_decoy_pocket
PDBscreen_HQ_cross_decoy_pocket             PDBscreen_LQ_active_pocket
##############################################################################################################################
'''
def filter_targets(file_paths,targets):
    filtered_keys = []
    for name in file_paths:
        if name.split('/')[-1].split('_')[0] in targets:
            filtered_keys.append(name)
    return filtered_keys
def get_part_data(data_dir,names,fast_num = 5,active_names = None):

    pro_decoy_pro = defaultdict(list)
    for i in names:
        pro = i.split('_')[0]
        if active_names is None:
            pro_decoy_pro[pro].append(i)
        else:
            if pro in active_names:
                pro_decoy_pro[pro].append(i)
    pro_decoy_pro_5 = defaultdict(list)
    for key in pro_decoy_pro.keys():
        if len(pro_decoy_pro[key]) <= fast_num:

            pro_decoy_pro_5[key] = pro_decoy_pro[key]
        else:
            pro_decoy_pro_5[key] =pro_decoy_pro[key][:fast_num]
    pro_decoy_pro_5 = sum(pro_decoy_pro_5.values(),[])
    print('mean of actives : decoys in cross_decoys',np.mean(list(dict(Counter([i.split('_')[2] for i in pro_decoy_pro_5])).values())))
    return [data_dir + name for name in pro_decoy_pro_5]
def get_part_data_screen_pose(data_dir,names,fast_num = 5,active_names = None):

    pro_decoy_pro = defaultdict(list)
    for i in names:
        pro = i.split('_active')[0]
        if active_names is None:
            pro_decoy_pro[pro].append(i)
        else:

            if pro in active_names:
                pro_decoy_pro[pro].append(i)

    pro_decoy_pro_5 = defaultdict(list)

    for key in pro_decoy_pro.keys():
        if len(pro_decoy_pro[key]) <= fast_num:
            pro_decoy_pro_5[key] = pro_decoy_pro[key]
        else:
            pro_decoy_pro_5[key] =pro_decoy_pro[key][:fast_num]
    pro_decoy_pro_5 = sum(pro_decoy_pro_5.values(),[])
    print('mean of actives : decoys in cross_decoys',np.mean(list(dict(Counter([i.split('_')[2] for i in pro_decoy_pro_5])).values())))

    return [data_dir + name for name in pro_decoy_pro_5]
def get_part_data_screen(data_dir,names,fast_num = 5,active_names = None):
    pro_decoy_pro = defaultdict(list)
    for i in names:
        pro = i.split('-')[0]
        if active_names is None:
            pro_decoy_pro[pro].append(i)
        else:
            if pro in active_names:
                pro_decoy_pro[pro].append(i)

    pro_decoy_pro_5 = defaultdict(list)

    for key in pro_decoy_pro.keys():
        if len(pro_decoy_pro[key]) <= fast_num:
            # print(key)

            pro_decoy_pro_5[key] = pro_decoy_pro[key]
        else:
            pro_decoy_pro_5[key] =pro_decoy_pro[key][:fast_num]
    pro_decoy_pro_5 = sum(pro_decoy_pro_5.values(),[])
    print('mean of actives : decoys in cross_decoys',np.mean(list(dict(Counter([i.split('_')[2].split('-')[1] for i in pro_decoy_pro_5])).values())))
    return [data_dir + name for name in pro_decoy_pro_5]
#------------------------------------------------------
# get HQ part from PDBscreen 
#------------------------------------------------------
valid_keys =glob.glob('../../training_data/PDBscreen/PDBscreen_HQ_active_pocket/*')
active_pros = set([v.split('/')[-1].split('_')[0] for v in valid_keys])

print('HQ actives pros :',len(active_pros))
cross_decoys= os.listdir('../../training_data/PDBscreen/PDBscreen_HQ_cross_decoy_pocket/')

cross_decoys_pros = set([v.split('/')[-1].split('_')[0] for v in cross_decoys])
print('cross_decoys pros :',len(cross_decoys_pros))
cross_decoys_dir = '../../training_data/PDBscreen/PDBscreen_HQ_cross_decoy_pocket/'

valid_keys += get_part_data(cross_decoys_dir,cross_decoys,fast_num=10,active_names = active_pros)

print('HQ len of decoys: ',len(cross_decoys))
#----------------------------------------------------------------
############# add pose enhanced samples ###############
cross_decoys_dir_pose = '../../training_data/PDBscreen/PDBscreen_HQ_active_augment_2a_pose_pocket/'
poses = os.listdir('../../training_data/PDBscreen/PDBscreen_HQ_active_augment_2a_pose_pocket/')
valid_keys_poses = get_part_data(cross_decoys_dir_pose,poses,fast_num=100,active_names = active_pros)
print('pose enhanced samples: ',len(valid_keys_poses))
print('pose pros: ',len(set([i.split('/')[-1].split('_')[0] for i in valid_keys_poses])))
valid_keys += valid_keys_poses
print('all valid keys in HQ ',len(valid_keys))
############################## add generated decoys ##############################
valid_keys_decoys = glob.glob('../../training_data/PDBscreen/PDBscreen_HQ_generated_decoy_pocket/*')
print('all Generated decoys : ',len(valid_keys_decoys))
valid_keys_decoys  = [i for i in valid_keys_decoys if i.split('/')[-1].split('_')[0] in active_pros]
print('last Generated decoys : ',len(valid_keys_decoys))
valid_keys += valid_keys_decoys
print('HQ len of generated decoys + active + crossdecoy  + pose enhanced : ',len(valid_keys))
######################################################################################
# get HQ data from PDBscreen
# then get LQ data from PDBscreen 
# but we can use it to train the model
# all data are we need to train the model
###########################################################################################################################
import pickle

with open('../other_info/pdbscreen_active_5_2.pkl','rb') as f:
    data_names = pickle.load(f)
data_names = [i.split('.')[0].replace('_active','') for i in data_names]
valid_keys_screen = glob.glob('../../training_data/PDBscreen/PDBscreen_LQ_active_pocket/*')
# filtering
valid_keys_screen = [ i for i in valid_keys_screen if i.split('/')[-1].replace('_active_0','') in data_names]
active_pros_screen = set([v.split('/')[-1].replace('_active_0','') for v in valid_keys_screen])

print('LQ actives pros :',len(active_pros_screen))
cross_decoys_screen= os.listdir('../../training_data/PDBscreen/PDBscreen_LQ_cross_decoy_pocket/')

cross_decoys_pros_screen = set([v.split('-')[0] for v in cross_decoys_screen])

print('LQ cross_decoys_pros :',len(cross_decoys_pros_screen))
cross_decoys_dir_screen = '../../training_data/PDBscreen/PDBscreen_LQ_cross_decoy_pocket/'

valid_keys_screen += get_part_data_screen(cross_decoys_dir_screen,cross_decoys_screen,fast_num=10,active_names = active_pros_screen)

print(' LQ len of decoys: ',len(cross_decoys_screen))
#-------------------------------------------------------------------------------
cross_decoys_dir_screen_pose = '../../training_data/PDBscreen/PDBscreen_LQ_active_augment_2a_pose_pocket/'
screen_poses = os.listdir('../../training_data/PDBscreen/PDBscreen_LQ_active_augment_2a_pose_pocket/')
active_pros_screen = [i.split('_')[0] + '_' +  i.split('_')[2] for i in active_pros_screen]
valid_screen_poses = get_part_data_screen_pose(cross_decoys_dir_screen_pose,screen_poses,fast_num=100,active_names =active_pros_screen)

print('pose enhanced LQ samples: ',len(valid_screen_poses))
print('LQ pose pros: ',len(set([i.split('/')[-1].split('_')[0] for i in valid_screen_poses])))
valid_keys_screen += valid_screen_poses
print('all valid keys in pdbbind ',len(valid_keys_screen))
########################################################### generated decoys ##################
valid_keys_screen_shape = glob.glob('../../training_data/PDBscreen/PDBscreen_LQ_generated_decoy_pocket/*')
print('LQ len of shape decoys : ',len(valid_keys_screen_shape))
# remove sp score > -5
data_names = [i.split('_')[0] + '_' +  i.split('_')[2] for i in data_names]
valid_keys_screen_shape = [i for i in valid_keys_screen_shape if i.split('/')[-1].split('_align')[0] in data_names]
print('LQ len of shape decoys  remove sp score > -5 kcal/mol: ',len(valid_keys_screen_shape))
valid_keys_screen_shape = [i for i in valid_keys_screen_shape if i.split('/')[-1].split('_align')[0] in active_pros_screen]

print(' LQ len of shape decoys : ',len(valid_keys_screen_shape))
valid_keys_screen += valid_keys_screen_shape
print(' LQ len of shape decoys + active + crossdecoy + pose enhanced: ',len(valid_keys_screen))

print('removeing duplicated target from training data ..........')
#------------------------------------------------------
valid_keys = valid_keys_screen + valid_keys
print('valid_keys_screen + valid_keys',len(valid_keys))


with open('../other_info/pdb_to_uniport_dict.pkl','rb') as f:
    pdb_to_uniport_dict = pickle.load(f)
with open('../other_info/uniport_to_pdb_dict.pkl','rb') as f:
    uniport_to_pdb_dict = pickle.load(f)
dups = ['4DJW', '1H1Q' ,'2GMX' ,'4HW3', '3FLY', '2QBS',' 2ZFF' ,'4GIH','5HNB' ,'4R1Y' ,'3L9H', '5TBM' ,'6HVI' ,'5EHR' ,'4PV0' ,'4UI5']
duplicated_targets = []
for dup in dups:
    if dup.lower() in pdb_to_uniport_dict.keys():
        uniport_id = pdb_to_uniport_dict[dup.lower()]
        duplicated_targets += uniport_to_pdb_dict[uniport_id]
duplicated_targets = set(duplicated_targets)
print(f'duplicated  pdbids################# {len(duplicated_targets)}')
dude_gene =  set(OrderedDict.fromkeys([v.split('/')[-1].split('_')[0] for v in valid_keys]))
print('len of the before remove duplicated target: ',len(dude_gene))
dude_gene = dude_gene - duplicated_targets
print('len of the after remove duplicated target: ',len(dude_gene))
print('remove done!')
#-----------------------------------------------------------------------
train_keys = [k for k in valid_keys if k.split('/')[-1].split('_')[0] in dude_gene]    
test_keys = [k for k in valid_keys if k.split('/')[-1].split('_')[0] in duplicated_targets]   
print ('Num train keys: ', len(train_keys))
print ('Num test keys: ', len(test_keys))
################## uniport hierarchy sample pipeline test ##############################
with open('../other_info/eight_class_remove_duplicated.pkl','rb') as f:
    eight_class_remove_duplicated = pickle.load(f)
with open('../other_info/uniport_to_pdb_dict.pkl','rb') as f:
    uniport_to_pdb_dict = pickle.load(f)
def select_uniport_for_val(eight_class_remove_duplicated,uniport_to_pdb_dict,rate = 0.12,seed = 42):
    np.random.seed(seed)
    val_uniport_ids = []
    val_uniport_pdb_ids =[]
    for uniport_set in eight_class_remove_duplicated:
        uniport_set = list(uniport_set)
        size = int(rate*len(uniport_set))
        val_uniport_ids.extend(np.random.choice(uniport_set,size = size))
    for uniport_id in val_uniport_ids:

        val_uniport_pdb_ids.extend(uniport_to_pdb_dict[uniport_id])
    return val_uniport_ids,val_uniport_pdb_ids

val_uniport_ids,val_uniport_pdb_ids = select_uniport_for_val(eight_class_remove_duplicated,uniport_to_pdb_dict,seed = 42)
val_keys = [k for k in train_keys if k.split('/')[-1].split('_')[0]  in val_uniport_pdb_ids ]
train_keys = [k for k in train_keys if k.split('/')[-1].split('_')[0] not in val_uniport_pdb_ids ]

print ('Num train keys: ', len(train_keys))
print ('Num val keys: ', len(val_keys))
print ('Num test keys: ', len(test_keys))
print('train/val = ',len(train_keys)/len(val_keys))
print('NUm uniport ids',len(val_uniport_ids))
with open('train_keys.pkl', 'wb') as f:
    pickle.dump(train_keys, f)
with open('test_keys.pkl', 'wb') as f:
    pickle.dump(test_keys, f)
with open('val_keys.pkl', 'wb') as f:
    pickle.dump(val_keys, f)