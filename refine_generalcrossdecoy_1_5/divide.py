import pickle
from collections import OrderedDict
import random
import glob
import numpy as np
import os
random.seed(0)
from collections import defaultdict,Counter
valid_keys = glob.glob('/home/caoduanhua/scorefunction/data/generalset_active_pocket_without_h/*')
valid_keys +=glob.glob('/home/caoduanhua/scorefunction/data/refineset_active_pocket_without_h/*')
# valid_keys +=glob.glob('/home/caoduanhua/score_function/data/dockingdecoy-bigger-10-caoduanhua-17w/*')
cross_decoys= os.listdir('/home/caoduanhua/scorefunction/data/generalset_refineset_crossdecoys_1_16_pocket_without_h/')
cross_decoys_dir = '/home/caoduanhua/scorefunction/data/generalset_refineset_crossdecoys_1_16_pocket_without_h/'
# print('len of actives: ',len(valid_keys))
# with open('/home/caoduanhua/scorefunction/GNN/GNN_graphformer/refine_generalcrossdecoy_1_10_keys/all_keys','rb') as f:
#     all_keys = pickle.load(f)
# cross_decoys = [cross_decoys_path + i for i in all_keys]
def get_part_data(data_dir,names,fast_num = 5):
    # data_name = []
    pro_decoy_pro = defaultdict(list)
    for i in names:
        pro_decoy_pro[i.split('_')[0]].append(i)

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

valid_keys += get_part_data(cross_decoys_dir,cross_decoys)
print('len of decoys: ',len(cross_decoys))
# print('decoy ')
# print('len of actives: ',len(valid_keys))
print('removeing duplicated target from training data ..........')
#------------------------------------------------------
with open('/home/caoduanhua/scorefunction/data/uniport_analysis/duplicated_with_independent_uniport_targets','rb') as f:
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