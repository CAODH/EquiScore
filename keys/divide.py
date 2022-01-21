import pickle
from collections import OrderedDict
import random
import glob
random.seed(0)
valid_keys = glob.glob('/home/caoduanhua/score_function/data/pocket_data/*')
valid_keys = [v.split('/')[-1] for v in valid_keys]
# print(valid_keys)
dude_gene =  list(OrderedDict.fromkeys([v.split('_')[0] for v in valid_keys]))
with open('./test.txt','r')as f:
    test_keys = f.readlines()
test_dude_gene = [key.strip() for key in test_keys]
train_dude_gene = [p for p in dude_gene if p not in test_dude_gene]
print(train_dude_gene)
train_keys = [k for k in valid_keys if k.split('_')[0] in train_dude_gene]    
test_keys = [k for k in valid_keys if k.split('_')[0] in test_dude_gene]    

print ('Num train keys: ', len(train_keys))
print ('Num test keys: ', len(test_keys))

with open('train_dude_gene.pkl', 'wb') as f:
    pickle.dump(train_dude_gene, f)
with open('test_dude_gene.pkl', 'wb') as f:
    pickle.dump(test_dude_gene, f)
with open('train_keys.pkl', 'wb') as f:
    pickle.dump(train_keys, f)
with open('test_keys.pkl', 'wb') as f:
    pickle.dump(test_keys, f)
    
