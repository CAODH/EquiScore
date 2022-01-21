#  caodunahua
# GNN_graphformer
GNN_DTI graphformer
# first check the ./keys/divide.py to make your train/test keys ,because  i only give a toy dataset to test my code !check the code and delete the train_keys = getToyKey(train_keys)
# test_keys = getToyKey(test_keys)  then add correct path to your data run divide.py 

# when you made your data in divide.py

# check ./utils.py GetEF func make sure in this func dont have gettoykey()

<!-- now you can run your data in this code !,if you want to debug this code ,you should ignore those tips  -->


<!-- now you can run this code  -->


#test
# python train.py --debug --epoch 1 --head_size 1 #add --debug to check code !

# follow paper setup
# python train.py 



#setup table
# since we are undirected edge so in_degree equal to out_degree  

———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————|
||                                               |
|  fundation_model          layer_type           |in_degree_bias     out_degree_bias    edge_bias          edge_type             rel_3d_pos_bias      rel_pos_bias       gate(only for Transformer_gate)
|________________________________________________|___________________________________________________________________________________________________________________________________________________________________________________________________________
|                           GAT_gate             |True or False       True or False     True or False   single or multi_hop       True or False      True or False False             True or False
|  paper                    MH_gate              |
|            Transformer_gate(select gate OR not)|          
|________________________________________________|____________________________________________________________________________________________________________________________________________________________________________________________________________
|                                                |
|graphformer               EncoderLayer          |            
|________________________________________________|_____________________________________________________________________________________________________________________________________________________________________________________________________________


#a sample for select a model with different attn_bias or attn_bias combines that you want add to selected model:

# python train.py --model 'paper'  --layer_type 'GAT_gate'  --rel_3d_pos_bias   # if you dont give --rel_3d_pos_bias in this command line all values have default,attn_bias's default all selected to False
also you can use :
# nohup python train.py --model 'paper'  --layer_type 'GAT_gate'  --rel_3d_pos_bias True > logfilename.log&

#test 

<!-- when you want to add a independent test give --test in command line,model will be test in args.test_path file(if dont give --test dafault false for speed in gridsearch.py) -->
sample:

# python train.py --model 'paper'  --layer_type 'GAT_gate'  --rel_3d_pos_bias True --test

also you can select different EF rate to test : default 0.01
# python train.py --model 'paper'  --layer_type 'GAT_gate'  --rel_3d_pos_bias True --test --EF_rate 0.01
# you also can give multi values for EF_rate like :--EF_rate 0.001,0.002,0.1,0.2
then will eval different EF_value 

# if you want to find the best model hyper-parameters or other optim ways
you can use gridsearch.py ;just write the hyper-parameters that you want to search in the main func loop!and set some hyper-parameter you dont want to fine-tuning 


# other value to setup
you can have some helpful ideas in train.py file! check the args plz!




