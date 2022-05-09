from layers import GetLayer
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
from multiprocessing import Pool
from layers import *
from graphformer_dataset import Batch
from torch.autograd import Variable
N_atom_features = 28
import copy
class gnn(torch.nn.Module):
    def __init__(self, args):
        super(gnn, self).__init__()
        self.args = args
        self.mu = nn.Parameter(torch.Tensor([args.initial_mu]).float())
        self.dev = nn.Parameter(torch.Tensor([args.initial_dev]).float())
        if args.auxiliary_loss:
            if args.deta_const:
                self.deta = 0.2
            else:
                self.deta = nn.Parameter(torch.Tensor([0.5]).float())
        if self.args.fundation_model  == 'paper':
            self.embede = nn.Linear(2*self.args.N_atom_features, self.args.n_in_feature, bias = False)
            #paper 在这里对特征做了映射；in_feature == d_graph_layer
        if self.args.fundation_model == 'graphformer':
            # print('+++++++++++++++++++++++++++++++++')
            atom_dim = 16*10 if self.args.FP else 10*5
            self.atom_encoder = nn.Embedding(atom_dim  + 1, self.args.n_out_feature, padding_idx=0)
        self.edge_encoder = nn.Embedding( 35* 5 + 1, self.args.head_size, padding_idx=0) if args.edge_bias is True else nn.Identity()
        self.rel_pos_encoder = nn.Embedding(512, self.args.head_size, padding_idx=0) if args.rel_pos_bias is True else nn.Identity()#rel_pos
        self.in_degree_encoder = nn.Embedding(10, self.args.n_out_feature, padding_idx=0) if args.in_degree_bias is True else nn.Identity()
        # self.out_degree_encoder = nn.Embedding(10, self.args.n_out_feature, padding_idx=0) if args.out_degree_bias is True else nn.Identity()
        self.rel_3d_encoder = nn.Embedding(65, self.args.head_size, padding_idx=0) if args.rel_3d_pos_bias is True else nn.Identity()
        #share layers
        self.layers1 = [self.args.n_out_feature for i in range(self.args.n_graph_layer+1)]
            #self.gconv1 = nn.ModuleList([GAT_gate(self.layers1[i], self.layers1[i+1]) for i in range(len(self.layers1)-1)]) 
        self.gconv1 = nn.ModuleList([GetLayer(args) for i in range(len(self.layers1)-1)]) #2层
        if not self.args.share_layer:
            self.gconv2 = nn.ModuleList([GetLayer(args) for i in range(len(self.layers1)-1)]) 
        self.FC = nn.ModuleList([nn.Linear(self.layers1[-1], self.args.d_FC_layer) if i==0 else
                                nn.Linear(self.args.d_FC_layer, 2) if i==self.args.n_FC_layer-1  else
                                nn.Linear(self.args.d_FC_layer, self.args.d_FC_layer) for i in range(self.args.n_FC_layer)]) #4层
        # two layer to regress for -logka/ki
        if self.args.add_logk_reg:
            self.FC_reg = nn.ModuleList([
                                    nn.Linear(self.args.d_FC_layer, 1) if i==self.args.n_FC_layer-3  else
                                    nn.Linear(self.args.d_FC_layer, self.args.d_FC_layer) for i in range(self.args.n_FC_layer-2)])
    def GetAttnBias(self,rel_pos,attn_bias,edge_input,all_rel_pos_3d_l, attn_edge_type,bias_one = False):

        # graph_attn_bias
        n_graph, n_node = attn_bias.size()[:2]#[bs,n,n]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(1, self.args.head_size, 1, 1) # [n_graph, n_head, n_node, n_node]

        # rel pos
        if self.args.rel_pos_bias is True:
            rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2)# # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            graph_attn_bias = graph_attn_bias + rel_pos_bias 
        # rel 3d pos
        if self.args.rel_3d_pos_bias is True and bias_one is  False:

            rel_3d_bias = self.rel_3d_encoder(all_rel_pos_3d_l).permute(0, 3, 1, 2)
            graph_attn_bias = graph_attn_bias + rel_3d_bias

        if self.args.edge_bias is True:
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2) # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            # print('edge_input in get attn bias: ',edge_input)
            graph_attn_bias = graph_attn_bias + edge_input
        if self.args.fundation_model == 'paper' and self.args.layer_type == 'GAT_gate' :
            graph_attn_bias = graph_attn_bias.sum(dim = 1)
        return graph_attn_bias
    def GetNodeFea(self,x,in_degree):
        # print('x: ',x.device)
        # print('embede weight: ',self.embede.weight.device)
        if self.args.fundation_model == 'paper':
            node_feature = self.embede(x)
        else:
            node_feature = self.atom_encoder(x.long()).mean(-2)
        if self.args.in_degree_bias:
            node_feature = node_feature + self.in_degree_encoder(in_degree)
        # if self.args.out_degree_bias:
        #     node_feature = node_feature + self.out_degree_encoder(out_degree)
        # self.out_degree_encoder(out_degree)
        return node_feature 
    def embede_graph_once(self, data,A2_limit):
        c_hs_1 = self.GetNodeFea(data.H,data.in_degree_1,data.out_degree_1)
        c_hs_2 = self.GetNodeFea(data.H,data.in_degree_2,data.out_degree_2)
        attn_bias_1 = self.GetAttnBias(data.rel_pos_1,data.attn_bias,data.edge_input_1,data.all_rel_pos_3d, data.attn_edge_type_1)
        attn_bias_2 = self.GetAttnBias(data.rel_pos_2,data.attn_bias,data.edge_input_2,data.all_rel_pos_3d, data.attn_edge_type_2)
        if self.args.only_adj2:
            data.A2 = torch.where(data.A2 > 0,1,0)
        elif self.args.only_dis_adj2:
            data.A2 = torch.where(data.A2 > self.mu,torch.exp(-torch.pow(data.A2-self.mu.expand_as(data.A2), 2)/(self.dev + 1e-6)),torch.tensor(1.0).to(self.dev.device))+ data.A1
        else:
            data.A2 = torch.exp(-torch.pow(data.A2-self.mu.expand_as(data.A2), 2)/self.dev + 1e-6) + data.A1

        if self.args.mode == '1_H' :
            for k in range(len(self.gconv1)):
                # if self.layer_type == 'GAT_gate':
                c_hs_1 = self.gconv1[k](c_hs_1, data.A1,self.args.use_adj,attn_bias_1)
                c_hs_2 = self.gconv1[k](c_hs_2, data.A2,self.args.use_adj,attn_bias_2)
            c_hs = c_hs_2-c_hs_1
            c_hs = F.dropout(c_hs, p=self.args.dropout_rate, training=self.training)
            c_hs = c_hs*data.V.unsqueeze(-1).repeat(1, 1, c_hs.size(-1))
            c_hs = c_hs.sum(1)
            return c_hs
    def embede_single_graph(self, data,A2_limit):
        c_hs_2 = self.GetNodeFea(data.H,data.in_degree_2,data.out_degree_2)
        attn_bias_2 = self.GetAttnBias(data.rel_pos_2,data.attn_bias,data.edge_input_2,data.all_rel_pos_3d, data.attn_edge_type_2)

        if A2_limit:#by caoduanhua
            for i ,item in enumerate(data.A2):
                n1,n2 = data.key[i]
                dm = item[:n1,n1:n2].clone()
                dm_t = item[n1:n2,:n1].clone()
            dm_adjust = torch.exp(-torch.pow(dm - self.mu.expand_as(dm),2)/self.dev)
            dm_adjust = torch.where(dm.float() < 5.0,dm_adjust.float(),torch.tensor(0.0).to(dm.device).float())
            dm_adjust_t = torch.exp(-torch.pow(dm_t - self.mu.expand_as(dm_t),2)/self.dev)
            dm_adjust_t = torch.where(dm_t.float() < 5.0,dm_adjust_t.float(),torch.tensor(0.0).to(dm_t.device).float())
            data.A2[i,:n1,n1:n2] = dm_adjust
            data.A2[i,n1:n2,:n1] = dm_adjust_t
        elif self.args.only_adj2:
            data.A2 = torch.where(data.A2 > 0,1,0)
        elif self.args.only_dis_adj2:
            data.A2 = torch.where(data.A2 > self.mu,torch.exp(-torch.pow(data.A2-self.mu.expand_as(data.A2), 2)/(self.dev + 1e-6)),torch.tensor(1.0).to(self.dev.device))+ data.A1
        else:
            data.A2 = torch.exp(-torch.pow(data.A2-self.mu.expand_as(data.A2), 2)/self.dev + 1e-6) + data.A1
        regularization = torch.empty(len(self.gconv1), device=data.H.device)
        for k in range(len(self.gconv1)):
            c_hs_2 = self.gconv1[k](c_hs_2, data.A2,self.args.use_adj,attn_bias_2)
            c_hs_2 = F.dropout(c_hs_2, p=self.args.dropout_rate, training=self.training)
        c_hs_2 = c_hs_2*data.V.unsqueeze(-1).repeat(1, 1, c_hs_2.size(-1))
        c_hs_2 = c_hs_2.sum(1)
        return c_hs_2
    
    def embede_graph(self, data,A2_limit):
        c_hs_1 = self.GetNodeFea(data.H,data.in_degree_1)
        c_hs_2 = self.GetNodeFea(data.H,data.in_degree_1)
        attn_bias_1 = self.GetAttnBias(data.rel_pos_1,data.attn_bias,data.edge_input_1,data.all_rel_pos_3d, data.attn_edge_type_1,bias_one = True)
        attn_bias_2 = self.GetAttnBias(data.rel_pos_1,data.attn_bias,data.edge_input_1,data.all_rel_pos_3d, data.attn_edge_type_1)
    
        if self.args.only_adj2:
            data.A2 = torch.where(data.A2 > 0,1,0)
            # data.A2 = data.A1
        elif self.args.only_dis_adj2:
            data.A2 = torch.where(data.A2 > self.mu,torch.exp(-torch.pow(data.A2-self.mu.expand_as(data.A2), 2)/(self.dev + 1e-6)),torch.tensor(1.0).to(self.dev.device))+ data.A1
        else:
            data.A2 = torch.exp(-torch.pow(data.A2-self.mu.expand_as(data.A2), 2)/self.dev + 1e-6) + data.A1
        # regularization = torch.empty(len(self.gconv1), device=data.H.device)
        if self.args.mode == '1_H' :
            c_hs = c_hs_1
            for k in range(len(self.gconv1)):
                # if self.layer_type == 'GAT_gate':
                c_hs_1 = self.gconv1[k](c_hs, data.A1,self.args.use_adj,attn_bias_1)
                c_hs_2 = self.gconv1[k](c_hs, data.A2,self.args.use_adj,attn_bias_2)
                c_hs = c_hs_2-c_hs_1
                c_hs = F.dropout(c_hs, p=self.args.dropout_rate, training=self.training)
            c_hs = c_hs*data.V.unsqueeze(-1).repeat(1, 1, c_hs.size(-1))
            c_hs = c_hs.sum(1)

            return c_hs         
        # return c_hs
    def GetATTMapOnce(self,data):
        c_hs_1 = self.GetNodeFea(data.H,data.in_degree_1,data.out_degree_1)
        c_hs_2 = self.GetNodeFea(data.H,data.in_degree_2,data.out_degree_2)
        attn_bias_1 = self.GetAttnBias(data.rel_pos_1,data.attn_bias,data.edge_input_1,data.all_rel_pos_3d, data.attn_edge_type_1)
        attn_bias_2 = self.GetAttnBias(data.rel_pos_2,data.attn_bias,data.edge_input_2,data.all_rel_pos_3d, data.attn_edge_type_2)

        if self.args.only_adj2:
            data.A2 = torch.where(data.A2 > 0,1,0)
        elif self.args.only_dis_adj2:
            data.A2 = torch.where(data.A2 > self.mu,torch.exp(-torch.pow(data.A2-self.mu.expand_as(data.A2), 2)/(self.dev + 1e-6)),torch.tensor(1.0).to(self.dev.device))+ data.A1
        else:
            data.A2 = torch.exp(-torch.pow(data.A2-self.mu.expand_as(data.A2), 2)/self.dev + 1e-6) + data.A1
        attention_list_1 = []
        attention_list_2 = []
        if self.args.mode == '1_H' :
            # c_hs = c_hs_1
            for k in range(len(self.gconv1)):
                # if self.layer_type == 'GAT_gate':
                c_hs_1,attention_1 = self.gconv1[k].GetAttentionMap(c_hs_1, data.A1,self.args.use_adj,attn_bias_1)
                c_hs_2,attention_2 = self.gconv1[k].GetAttentionMap(c_hs_2, data.A2,self.args.use_adj,attn_bias_2)
                attention_list_1.append(attention_1)
                attention_list_2.append(attention_2)
            c_hs = c_hs_2-c_hs_1
            c_hs = F.dropout(c_hs, p=self.args.dropout_rate, training=self.training)
            # c_hs = c_hs*data.V.unsqueeze(-1).repeat(1, 1, c_hs.size(-1))
            # c_hs = c_hs.sum(1)

            return attention_list_1,attention_list_2
    def GetATTMap(self,data):
        c_hs_1 = self.GetNodeFea(data.H,data.in_degree_1,data.out_degree_1)
        c_hs_2 = self.GetNodeFea(data.H,data.in_degree_2,data.out_degree_2)
        attn_bias_1 = self.GetAttnBias(data.rel_pos_1,data.attn_bias,data.edge_input_1,data.all_rel_pos_3d, data.attn_edge_type_1)
        attn_bias_2 = self.GetAttnBias(data.rel_pos_2,data.attn_bias,data.edge_input_2,data.all_rel_pos_3d, data.attn_edge_type_2)

        if self.args.only_adj2:
            data.A2 = torch.where(data.A2 > 0,1,0)
        elif self.args.only_dis_adj2:
            data.A2 = torch.where(data.A2 > self.mu,torch.exp(-torch.pow(data.A2-self.mu.expand_as(data.A2), 2)/(self.dev + 1e-6)),torch.tensor(1.0).to(self.dev.device))+ data.A1
        else:
            data.A2 = torch.exp(-torch.pow(data.A2-self.mu.expand_as(data.A2), 2)/self.dev + 1e-6) + data.A1
        attention_list_1 = []
        attention_list_2 = []
        if self.args.mode == '1_H' :
            c_hs = c_hs_1
            for k in range(len(self.gconv1)):
                # if self.layer_type == 'GAT_gate':
                c_hs_1,attention_1 = self.gconv1[k].GetAttentionMap(c_hs, data.A1,self.args.use_adj,attn_bias_1)
                c_hs_2,attention_2 = self.gconv1[k].GetAttentionMap(c_hs, data.A2,self.args.use_adj,attn_bias_2)
                attention_list_1.append(attention_1)
                attention_list_2.append(attention_2)
                c_hs = c_hs_2-c_hs_1
                c_hs = F.dropout(c_hs, p=self.args.dropout_rate, training=self.training)
            # c_hs = c_hs*data.V.unsqueeze(-1).repeat(1, 1, c_hs.size(-1))
            # c_hs = c_hs.sum(1)

            return attention_list_1,attention_list_2
    def reg_head(self,c_hs):
        for k in range(len(self.FC_reg)):
            if k<len(self.FC_reg)-1:
                c_hs = self.FC_reg[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.args.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
                # print(c_hs.shape)
            else:
                c_hs = self.FC_reg[k](c_hs)
        # print(c_hs.shape)
        return c_hs
    def fully_connected(self, c_hs):
        # regularization = torch.empty(len(self.FC)*1-1, device=c_hs.device)

        for k in range(len(self.FC)):
          
            if k<len(self.FC)-1:
                if k==len(self.FC)-3 and self.args.add_logk_reg:
                    self.reg_value = self.reg_head(c_hs)
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.args.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.FC[k](c_hs)
        if  self.args.r_drop  or self.args.loss_fn == 'focal_loss' or self.args.loss_fn == 'cross_entry': 
            return c_hs
        elif self.args.loss_fn == 'mse_loss':
            return c_hs[:,1]
        else:

            try:
                c_hs = torch.sigmoid(c_hs[:,1])
            except:
                c_hs = torch.sigmoid(c_hs)
            return c_hs
            # return c_hs

        # c_hs = torch.softmax(c_hs,dim = -1)

        # return c_hs
    #add by caodunahua for distribute training
    def forward(self,A2_limit,sample):
        #embede a graph to a vector
        if self.args.embed_graph == 'single_graph':
            c_hs = self.embede_single_graph(sample,A2_limit)
        elif self.args.embed_graph == 'embed_graph_once':
            c_hs = self.embede_graph_once(sample,A2_limit)
        elif self.args.embed_graph == 'double_graph':
            c_hs = self.embede_graph(sample,A2_limit)
        else:
            raise ValueError('not implement thie kind embed graph method !')
        #fully connected NN
        c_hs = self.fully_connected(c_hs)
        # print('c_hs after ebed_graph: ',c_hs)
        if self.args.loss_fn == 'bce_loss' and not self.args.r_drop:
            c_hs = c_hs.view(-1) 
        #note that if you don't use concrete dropout, regularization 1-2 is zero
        return c_hs
    # metamix setup functions

