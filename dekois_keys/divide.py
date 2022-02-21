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
    return [data_dir + name for name in pro_decoy_pro_5]
valid_keys = glob.glob('/home/caoduanhua/score_function/data/general_refineset/challenge_decoy_filter_top20/*')
valid_keys +=glob.glob('/home/caoduanhua/score_function/data/general_refineset/generalset_active_pocket_without_h/*')
valid_keys +=glob.glob('/home/caoduanhua/score_function/data/general_refineset/refineset_active_pocket_without_h/*')
active_pros = set([v.split('/')[-1].split('_')[0] for v in valid_keys])

print('actives pros :',len(active_pros))
cross_decoys= os.listdir('/home/caoduanhua/score_function/data/general_refineset/generalset_refineset_crossdecoys_1_16_pocket_without_h/')
cross_decoys_pros = set([v.split('/')[-1].split('_')[0] for v in cross_decoys])
print('cross_decoys_pros :',len(cross_decoys_pros))
cross_decoys_dir = '/home/caoduanhua/score_function/data/general_refineset/generalset_refineset_crossdecoys_1_16_pocket_without_h/'

valid_keys += get_part_data(cross_decoys_dir,cross_decoys,fast_num=5,active_names = active_pros)

print('len of decoys: ',len(cross_decoys))
# print('decoy ')
# print('len of actives: ',len(valid_keys))
print('removeing duplicated target from training data ..........')
#------------------------------------------------------
with open('/home/caoduanhua/score_function/data/uniport_analysis/duplicated_with_dekois_independent_uniport_targets','rb') as f:
    duplicated_targets = pickle.load(f)
    print('duplicated tragets: ',len(duplicated_targets))
dude_gene =  set(OrderedDict.fromkeys([v.split('/')[-1].split('_')[0] for v in valid_keys]))
print('len of the before remove duplicated target: ',len(dude_gene))
dude_gene = dude_gene - duplicated_targets
print('len of the after remove duplicated target: ',len(dude_gene))
print('remove done!')
#-----------------------------------------------------------------------
# valid_keys = [v.split('/')[-1] for v in valid_keys]
# print(valid_keys)
# dude_gene =  list(OrderedDict.fromkeys([v.split('/')[-1].split('_')[0] for v in valid_keys]))

# with open('./test.txt','r')as f:
#     test_keys = f.readlines()
# test_dude_gene = [key.strip() for key in test_keys]
# train_dude_gene = [p for p in dude_gene if p not in test_dude_gene]
# print(train_dude_gene)
train_keys = [k for k in valid_keys if k.split('/')[-1].split('_')[0] in dude_gene]    
test_keys = [k for k in valid_keys if k.split('/')[-1].split('_')[0] in duplicated_targets]   
print ('Num train keys: ', len(train_keys))
print ('Num test keys: ', len(test_keys))

with open('train_keys.pkl', 'wb') as f:
    pickle.dump(train_keys, f)
with open('test_keys.pkl', 'wb') as f:
    pickle.dump(test_keys, f)