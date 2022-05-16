import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
import torch
'''
paper 
paper_with_A2_limit
paper_with_multi_head
paper_with_multi_head_A2_limit
paper_with_transformer

paper_with_transformer_A2_limit
paper_with_transformer_gate
config should have :
fundation module:[paper,graphformer]
feature extractor:[GAT,MH,Transformer]
A2_limit:[True,False]
gate:[True,False]#ongly in transformer_gate 

attn_bias = [3d_distance,edge,degree,center,angle]

怎么提高可拓展性？
'''

#pure transformer layer and graphformer layer! 这里输入需要时经过和GAT_gate 不同的编码层

class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.args = args
        self.self_attention_norm = GetGrpahNorm(self.args.n_out_feature,self.args.norm_type)
        self.self_edge_norm = GetGrpahNorm(self.args.edge_dim,norm_type = 'ln')

        self.self_attention = GetAttMode(self.args)
        self.self_attention_dropout = nn.Dropout(self.args.attention_dropout_rate)
        self.ffn_norm = GetGrpahNorm(self.args.n_out_feature,self.args.norm_type)
        self.ffn = FeedForwardNetwork(self.args.n_out_feature, self.args.ffn_size, self.args.dropout_rate)
        #edge projet 
        self.ffn_dropout_edge = nn.Dropout(self.args.dropout_rate)
        self.ffn_edge_norm = GetGrpahNorm(self.args.edge_dim,norm_type = 'ln')
        self.ffn_edge = FeedForwardNetwork(self.args.edge_dim, self.args.ffn_size, self.args.dropout_rate)

        self.ffn_dropout = nn.Dropout(self.args.dropout_rate)
        # self.ffn_dropout_edge = nn.Dropout(self.args.dropout_rate)

    def forward(self, x, adj,adj1,use_adj,edge_fea,attn_bias=None):#不同的注意力的偏执测试
        y = self.self_attention_norm(x)
        edge_fea_norm = self.self_edge_norm(edge_fea)

        y,edge_fea_norm = self.self_attention(y, y, y, adj,adj1,use_adj,edge_fea_norm,attn_bias)

        # edge layer module
        edge_fea_norm = self.ffn_dropout_edge(edge_fea_norm)
        edge_fea_norm = edge_fea + edge_fea_norm
        edge_fea_norm = self.ffn_edge_norm(edge_fea_norm)
        edge_fea_norm = self.ffn_edge(edge_fea_norm)
        edge_fea_norm = self.ffn_dropout_edge(edge_fea_norm)
        edge_fea = edge_fea + edge_fea_norm

        # x layer module
        y = self.self_attention_dropout(y)
        x = x + y
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x,edge_fea

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size,edge_dim):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size
        assert hidden_size % head_size == 0
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.att_dropout_local = nn.Dropout(attention_dropout_rate)
        
        # edge output project
        # self.attn = torch.nn.Sequential()
        self.attn_project = nn.Linear(head_size, edge_dim)
        self.edge_project = nn.Linear(edge_dim, head_size * 1)
        ############################# out put project
        self.output_layer = nn.Linear(head_size * att_size, hidden_size)
        self.output_layer_edge = nn.Linear(edge_dim, edge_dim,bias=False)

    def forward(self, q, k, v, adj,adj1,use_adj,edge_fea,attn_bias=None,):
        orig_q_size = q.size()
        # [bs,n,5，8] 
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        #[bs,n,5] > [bs,n,5，8] > [bs,-1,8,140/8]
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)
        # print('q.shape:',q.shape)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if use_adj:
            # adj = torch.where(adj > 0,1,0)
            zero_vec = -9e15*torch.ones_like(x)
            x = torch.where(adj.unsqueeze(1) > 0, x, zero_vec)
            x = x*adj.unsqueeze(1)
        if attn_bias is not None:
            # print('MH ATT: ',x.shape,attn_bias.shape)
            # gated attn
            x_long = x*attn_bias

        # get long-range interaction
        x_long = F.softmax(x_long, dim=-1)
        x_long = self.att_dropout(x_long)
        x_long = x_long.matmul(v)  # [b, h, q_len, attn]
        # get local interaction

        # add edge attn bias and project edge fea
        edge_bias = self.edge_project(edge_fea).transpose(1,3).contiguous()# [bs,heads,n,n]

        # edge get node information
       
        x_project = self.attn_project(x.transpose(1,3).contiguous())# [bs,n,n,hiddensize]
        # only get adj1== 1 informations
        x_project = x_project.transpose(1,3).contiguous()*adj1.unsqueeze(1) # adj must ba the adj1 
        edge_fea = edge_fea +  x_project.transpose(1,3).contiguous()

        edge_fea = self.output_layer_edge(edge_fea)

        x_local  = x * edge_bias*adj1.unsqueeze(1)

        x_local = F.softmax(x_local, dim=-1)
        # x = torch.softmax(x, dim=-1)
        x_local = self.att_dropout_local(x_local)
        x_local = x_local.matmul(v)  # [b, h, q_len, attn]
        # add long_range local_range interaction
        x = x_local + x_long
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x,edge_fea
class DistangledMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(DistangledMultiHeadAttention, self).__init__()

        self.head_size = head_size
        assert hidden_size % head_size == 0
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.linear_unary = nn.Linear(hidden_size,head_size)
        
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, adj,use_adj,attn_bias=None):
        orig_q_size = q.size()
        # [bs,n,5，8] 
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        #construct distangled atttention as unary(k_len)
        unary = self.linear_unary(k).transpose(1,2).contiguous()  # [batch, k_len, num_heads] -> [batch, num_heads, k_len]
        unary = unary.view(batch_size, self.head_size,1,-1)#[batch, num_heads, 1,k_len]

        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)
        # print('q.shape:',q.shape)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
        # construct distangled atttion as exp(q-mean_e)@(k - mean_k) + exp(unary(k))
        mean_q = torch.mean(q,dim = -2,keepdim = True)
        mean_k = torch.mean(k,dim = -1,keepdim = True)
        q -= mean_q
        k -= mean_k
        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x= torch.matmul(q, k)  # [b, h, q_len, k_len]
    
        # x = unary + x

        if use_adj:
            zero_vec = -9e15*torch.ones_like(x)
            x = torch.where(adj.unsqueeze(1) > 0, x, zero_vec)
            x = F.softmax(x, dim=-1)
            unary= F.softmax(unary, dim=-1)
            x= unary + x
            x = x*adj.unsqueeze(1)
        else:
            x = F.softmax(x, dim=-1)
            unary= F.softmax(unary, dim=-1)
            x= unary + x
        if attn_bias is not None:
            x = x + attn_bias
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x

import torch
import torch.nn as nn
class Norm_mul(nn.Module):
    def __init__(self, norm_type, hidden_dim=64, print_info=None):
        super(Norm_mul, self).__init__()
        # assert norm_type in ['bn', 'ln', 'gn', None]
        self.norm = None
        self.print_info = print_info
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
    def forward(self,tensor, print_=False):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor
        mean_tensor = tensor.mean(dim= 1).unsqueeze(1).expand_as(tensor)
        std = torch.mean((tensor - self.mean_scale*mean_tensor).pow(2),dim = 1)
        std = std.sqrt().unsqueeze(1) 
        tensor = (self.weight*(tensor - self.mean_scale*mean_tensor)/(std + 1e-6) + self.bias)
        return tensor
class Norm_iter(nn.Module):

    def __init__(self, norm_type, hidden_dim=64, print_info=None):
        super(Norm_iter, self).__init__()
        # assert norm_type in ['bn', 'ln', 'gn', None]
        self.norm = None
        self.print_info = print_info
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self,tensor, print_=False):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor
        new_tensor = torch.zeros_like(tensor).to(tensor.device).float()
        # print(new_tensor.requires_grad)
        for i,graph in enumerate(tensor):
            #对每个图求graphnorm
            row_sum = torch.sum(graph,dim = -1)
            # print(row_sum)
            num_nodes = torch.sum(row_sum != 0)
            # print(num_nodes)
            mean = torch.sum(graph,dim = 0)/num_nodes
            #计算方差
            std = torch.mean((graph[:num_nodes] - self.mean_scale*mean).pow(2),dim = 0).sqrt()
            # print('std:',std)
            new_tensor[i][:num_nodes] = self.weight*(graph[:num_nodes]-self.mean_scale*mean)/(std + 1e-6)+ self.bias 
        # print(new_tensor.requires_grad)
        return new_tensor
def GetGrpahNorm(hidden_size,norm_type = 'gn_mul'):
    if norm_type == 'gn_mul':
        return Norm_mul('gn',hidden_size)
    elif norm_type == 'gn_iter':
        return Norm_iter('gn',hidden_size)
    elif norm_type == 'ln':
        return nn.LayerNorm(hidden_size)
    else:
        raise ValueError('not support this mode norm mode! check the code plz!')
def GetAttMode(args):
    if args.att_mode == 'DSA':
        return DistangledMultiHeadAttention(args.n_out_feature, args.attention_dropout_rate, args.head_size,args.edge_dim)
    elif args.att_mode == 'SA':
        return MultiHeadAttention(args.n_out_feature, args.attention_dropout_rate, args.head_size,args.edge_dim)
    else:
        raise ValueError('not support this mode attention! until now! check your param')
def GetLayer(args):
    if args.fundation_model == 'graphformer':
        return EncoderLayer(args)