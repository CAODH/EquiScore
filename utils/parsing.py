from argparse import ArgumentParser,FileType
def parse_train_args():
    # General arguments
    parser = ArgumentParser()
    # parser.add_argument('--config', type=FileType(mode='r'), default=None)
    parser.add_argument('--save_dir', type=str, default='workdir', help='Folder in which to save model and logs')
    parser.add_argument('--local_rank', type=int, default = -1,help='local rank for DistributedDataParallel')
    parser.add_argument('--save_model', type=str, default = './workdir/official_weight/save_model_screen.pt',help='model for test or restart training')
    parser.add_argument('--lmdb_cache', type=str, default='./lmdbs/PDBscreen', help='Folder containing trainging data in lmdb format')
    parser.add_argument('--data_path', type=str, default='./EquiScore/data/training_data/PDBscreen', help='Folder containing trainging data')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode for EquiScore')
    parser.add_argument('--hot_start', action='store_true', default=False, help='Continue training on the basis of the original please set save_model to continue training')
    # Dataset
    parser.add_argument('--train_keys', type=str, default='./data/data_splits/screen_model/train_keys.pkl', help='Path of file defining the split')
    parser.add_argument('--val_keys', type=str, default='./data/data_splits/screen_model/val_keys.pkl', help='Path of file defining the split')
    parser.add_argument('--test_keys', type=str, default='./data/data_splits/screen_model/test_keys.pkl', help='Path of file defining the split')
    parser.add_argument('--train_val_mode', type=str, default='uniport_cluster', help='data split mode')
    # parser.add_argument('--test_keys', type=str, default='./data/test_keys', help='Path of file defining the split')
    # distributed training
    parser.add_argument('--ngpu', type=int, default=1, help='Number of gpu for training')
    parser.add_argument('--MASTER_ADDR', type=str, default='localhost', help='localhost or ip address')
    parser.add_argument('--MASTER_PORT', type=str, default='29505', help='port number')
    # Training arguments
    parser.add_argument('--epoch', type=int, default=400, help='Number of epochs for training')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--loss_fn', type=str, default='cross_entry',\
                        choices=['bce_loss','focal_loss','cross_entry','mse_loss','poly_loss_ce','poly_loss_fl'], help='loss function')
    parser.add_argument('--grad_sum', type=int, default=1, help='Number of grad accumulation steps')
    parser.add_argument('--label_smothing', type=float, default=0.0, help='label smothing coefficient')
    parser.add_argument('--eps', type=float, default=4.0, help='focal loss eps')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size on each gpu')
    parser.add_argument('--sampler', action='store_true', default=False, help='dynamic sampler or not')
    parser.add_argument('--scheduler', type=str, default=None, help='LR scheduler')
    parser.add_argument('--patience', type=int, default=50, help='Patience of early stopping')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--lr_decay', action='store_true', default=True, help='dynamic sampler or not')
    parser.add_argument('--max_lr', type=float, default=0.001, help='max learning rate') 
    parser.add_argument('--pct_start', type=float, default=0.3, help='OneCycleLR parameter')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for preprocessing')
    # Model
    parser.add_argument('--model', type=str, default='EquiScore', help='model class')
    parser.add_argument('--pred_mode', type=str, default='ligand', help='prediction mode')
    parser.add_argument('--n_graph_layer', type=int, default=2, help='Number of EquiScore layers')
    parser.add_argument('--threshold', type=float, default=5.5, help='Radius cutoff for geometric diatance based graph')
    parser.add_argument('--n_FC_layer', type=int, default=4, help='Number of linear layers')
    parser.add_argument('--d_FC_layer', type=int, default=256, help='dims of linear layers')
    parser.add_argument('--n_in_feature', type=int, default=128, help='dims of input features in EquiScore')
    parser.add_argument('--n_out_feature', type=int, default=128, help='dims of output features in EquiScore')
    parser.add_argument('--ffn_size', type=int, default=280, help='dims of FFN layers in EquiScore')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout_rate in EquiScore')
    parser.add_argument('--attention_dropout_rate', type=float, default=0.1, help='dropout_rate in Attention module')
    parser.add_argument('--head_size', type=int, default=8, help='number of attention heads in EquiScore')
    parser.add_argument('--pos_enc_dim', type=int, default=16, help='position encoding dims')
    parser.add_argument('--edge_dim', type=int, default=64, help='edge feature dims')
    parser.add_argument('--layer_norm', action='store_true', default=True, help='layer norm or not')
    parser.add_argument('--graph_norm', action='store_true', default=True, help='graph norm on node features or not')
    parser.add_argument('--residual', action='store_true', default=True, help='residual connect or not')
    parser.add_argument('--edge_bias', action='store_true', default=True, help='covalent bond informations')
    parser.add_argument('--rel_3d_pos_bias', action='store_true', default=True, help='3d distance informations')
    parser.add_argument('--in_degree_bias', action='store_true', default=True, help='in degree infomations')
    parser.add_argument('--virtual_aromatic_atom', action='store_true', default=True, help='add virtual aromatic atom')
    parser.add_argument('--fingerprintEdge', action='store_true', default=True, help='construct edge based fingerprint information by proLIF ')
    parser.add_argument('--FP', action='store_true', default=True, help='use AttentiveFP feature or graphformer feature')
    parser.add_argument('--rel_pos_bias', action='store_true', default=False, help=' shortest path distance informations')
    parser.add_argument('--lap_pos_enc', action='store_true', default=False, help='laplace position infomations')
    # Benchmark test and Screen compounds for a target protein
    parser.add_argument('--test', action='store_true', default=False, help='test mode for banchmark test or screen compounds')
    parser.add_argument('--test_mode', type = str,default='one_pose', choices=['one_pose','multi_pose'],help="if dcoking result one pose for a ligand set 'one_pose' or multi poses for a ligand set multi_pose")
    parser.add_argument('--test_flag', type = str,default='external_test',help="anything can help you to identify the test result")
    parser.add_argument('--idx_style', action='store_true', default=False, help='for multi_pose mode to select multi pose test or one pose test in the pose number')
    parser.add_argument('--pose_num', type=int, default=1, \
                        help='select the pose number for multi_pose mode test ,for example : \
                            if pose_num = 3, idx_style=true, just the 3rd pose for test,\
                            but if idx_style=false, the first 3 poses will be test')
    parser.add_argument('--test_path', type=str, default="./data/sample_data/", \
                        help='test directory which contains the banchmark test data dirs')
    parser.add_argument('--test_name', type=str, default="tmp_pockets", \
                        help='test dataset directory which contains the test pockets')
    parser.add_argument('--EF_rates', type=list, default=[0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5], \
                        help='EF rates for banchmark test')
    parser.add_argument('--pred_save_path', type=str, default="./data/test_results/EquiScore_pred_for_tmp_pockets.pkl", \
                        help='path for prediction results in Screening script')
    args = parser.parse_args()
    return args
