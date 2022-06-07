import pickle
from collections import OrderedDict
import random
import glob
import numpy as np
import os
random.seed(0)
from collections import defaultdict,Counter
def filter_targets(file_paths,targets):
    filtered_keys = []
    for name in file_paths:
        if name.split('/')[-1].split('_')[0] in targets:
            filtered_keys.append(name)
    return filtered_keys

def get_part_data(data_dir,names,fast_num = 5,active_names = None):
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
    for key in pro_decoy_pro.keys():
        if len(pro_decoy_pro[key]) <= fast_num:
            # print(key)

            pro_decoy_pro_5[key] = pro_decoy_pro[key]
        else:
            pro_decoy_pro_5[key] =pro_decoy_pro[key][:fast_num]
    pro_decoy_pro_5 = sum(pro_decoy_pro_5.values(),[])
    print('mean of actives : decoys in cross_decoys',np.mean(list(dict(Counter([i.split('_')[2] for i in pro_decoy_pro_5])).values())))
    return [os.path.join(data_dir , name) for name in pro_decoy_pro_5]
def get_part_data_screen(data_dir,names,fast_num = 5,active_names = None):
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
    # print(' pro_decoy_pro_5',len( pro_decoy_pro_5))
    for key in pro_decoy_pro.keys():
        if len(pro_decoy_pro[key]) <= fast_num:
            # print(key)

            pro_decoy_pro_5[key] = pro_decoy_pro[key]
        else:
            pro_decoy_pro_5[key] =pro_decoy_pro[key][:fast_num]
    pro_decoy_pro_5 = sum(pro_decoy_pro_5.values(),[])
    print('mean of actives : decoys in cross_decoys',np.mean(list(dict(Counter([i.split('_')[2].split('-')[1] for i in pro_decoy_pro_5])).values())))
    # print('all decoy ',len(pro_decoy_pro_5)/len(active_names))
    return [os.path.join(data_dir , name) for name in pro_decoy_pro_5]
# get pose enhanced data
def get_part_data_pose(data_dir,names,fast_num = 5,active_names = None,mode = 'pdb'):
    # data_name = []
    pro_decoy_pro = defaultdict(list)
    for i in names:
        if mode=='pdb':
            pro = '_'.join(i.split('_')[:1])
        elif mode=='screen':
            pro = '_'.join(i.split('_')[:2])
        else:
            raise ValueError('not implement this mode')
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
    keys = pro_decoy_pro_5.keys()
    pro_decoy_pro_5 = sum(pro_decoy_pro_5.values(),[])
    print('mean of actives : decoys in cross_decoys',len(pro_decoy_pro_5)/(len(active_pros) if active_pros else len(keys)))
    return [os.path.join(data_dir , name) for name in pro_decoy_pro_5]
#------------------------------------------------------
# 先获取了pdbbind 的数据

# valid_keys = glob.glob('/home/caoduanhua/score_function/data/pdb_screen/active_pocket/*')
valid_keys =glob.glob('/home/caoduanhua/score_function/data/general_refineset/refineset_active_pocket_without_h/*')
valid_keys +=glob.glob('/home/caoduanhua/score_function/data/general_refineset/generalset_active_pocket_without_h/*')
active_pros = set([v.split('/')[-1].split('_')[0] for v in valid_keys])

print('pdb actives pros :',len(active_pros))
cross_decoys= os.listdir('/home/caoduanhua/score_function/data/general_refineset/generalset_refineset_crossdecoys_1_16_pocket_without_h/')

cross_decoys_pros = set([v.split('/')[-1].split('_')[0] for v in cross_decoys])
print('cross_decoys_pros :',len(cross_decoys_pros))
cross_decoys_dir = '/home/caoduanhua/score_function/data/general_refineset/generalset_refineset_crossdecoys_1_16_pocket_without_h/'

valid_keys += get_part_data(cross_decoys_dir,cross_decoys,fast_num=5,active_names = active_pros)

print(' pdbbind len of decoys: ',len(cross_decoys))
# pose data############################################
pose_active_dir = '/home/caoduanhua/score_function/data/general_refineset/pdbbind-data_arguement_active_2a_3_pocket-4/'
pose_actives = os.listdir(pose_active_dir)
pose_actives = get_part_data_pose(pose_active_dir,pose_actives,fast_num = 4,active_names = active_pros,mode = 'pdb')
print(' active pose num : ',len(pose_actives))
valid_keys +=pose_actives
pose_decoy_dir = '/home/caoduanhua/score_function/data/general_refineset/pdbbind-data_arguement_decoy_6a_7_pocket-7/'
pose_decoys = os.listdir(pose_decoy_dir)
pose_decoys = get_part_data_pose(pose_decoy_dir,pose_decoys ,fast_num = 7,active_names = active_pros,mode = 'pdb')
print(' active pose num : ',len(pose_decoys))
valid_keys +=pose_decoys
#----------------------------------------------------------------
# 先获取了pdbbind 的数据
# import pickle
# with open('/home/caoduanhua/score_function/data/pdb_screen/score_bins_name_file/pdbscreen_active_5_2.pkl','rb') as f:
#     data_names = pickle.load(f)
# data_names = [i.split('.')[0].replace('_active','') for i in data_names]

# valid_keys_screen = glob.glob('/home/caoduanhua/score_function/data/pdb_screen/active_pocket/*')
# # 过滤

# valid_keys_screen = [ i for i in valid_keys_screen if i.split('/')[-1].replace('_active_0','') in data_names]
# active_pros_screen = set([v.split('/')[-1].replace('_active_0','') for v in valid_keys_screen])

# print('pdb screen actives pros :',len(active_pros_screen))
# cross_decoys_screen= os.listdir('/home/caoduanhua/score_function/data/pdb_screen/cross_decoy_pocket/')

# cross_decoys_pros_screen = set([v.split('-')[0] for v in cross_decoys_screen])

# print('cross_decoys_pros :',len(cross_decoys_pros_screen))
# cross_decoys_dir_screen = '/home/caoduanhua/score_function/data/pdb_screen/cross_decoy_pocket/'

# valid_keys_screen += get_part_data_screen(cross_decoys_dir_screen,cross_decoys_screen,fast_num=6,active_names = active_pros_screen)

# print(' pdd screen len of decoys: ',len(cross_decoys_screen))
# #-------------------------------------------------------------------------------
# # ##################pose data ############################################
# pose_active_pros = active_pros_screen = set([v.split('/')[-1].split('_')[0] + '_' +  v.split('/')[-1].split('_')[2] for v in valid_keys_screen])
# screen_pose_active_dir = '/home/caoduanhua/score_function/data/pdb_screen/pdb_sceen_active_2a_4_pocket/'
# screen_pose_actives = os.listdir(screen_pose_active_dir)
# screen_pose_actives = get_part_data_pose(screen_pose_active_dir,screen_pose_actives,fast_num = 4,active_names = pose_active_pros ,mode = 'screen')
# print(' active pose num : ',len(screen_pose_actives))
# valid_keys_screen +=screen_pose_actives
# screen_pose_decoy_dir = '/home/caoduanhua/score_function/data/pdb_screen/pdb_sceen_decoy_6a_7_pocket/'
# screen_pose_decoys = os.listdir(screen_pose_decoy_dir)
# screen_pose_decoys = get_part_data_pose(screen_pose_decoy_dir,screen_pose_decoys ,fast_num = 7,active_names = pose_active_pros ,mode = 'screen')
# print(' active pose num : ',len(screen_pose_decoys))
# valid_keys_screen +=screen_pose_decoys
# #----------------------------------------------------------------

# print('removeing duplicated target from training data ..........')
# #------------------------------------------------------
# valid_keys = valid_keys_screen + valid_keys
print('valid_keys_screen + valid_keys',len(valid_keys))

with open('/home/caoduanhua/score_function/data/uniport_analysis/duplicated_to_pdb_screen_pdbbind_targets/duplicated_with_all_independent_uniport_targets','rb') as f:
    duplicated_targets = pickle.load(f)
    print('duplicated tragets: ',len(duplicated_targets))
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
with open('train_keys.pkl', 'wb') as f:
    pickle.dump(train_keys, f)
with open('test_keys.pkl', 'wb') as f:
    pickle.dump(test_keys, f)