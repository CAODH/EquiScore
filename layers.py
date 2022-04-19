import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import time
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
class GAT_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GAT_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        #self.A = nn.Parameter(torch.Tensor(n_out_feature, n_out_feature))
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_out_feature*2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj,use_adj,attn_bias=None):
        h = self.W(x)
        batch_size = h.size()[0]
        N = h.size()[1]
        e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h,self.A), h))
        e = e + e.permute((0,2,1))
        if use_adj:

            zero_vec = -9e15*torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            #attention = F.dropout(attention, self.dropout, training=self.training)
            #h_prime = torch.matmul(attention, h)
            attention = attention*adj
        if attn_bias is not None:
            attention += attn_bias
        # attention = F.softmax(attention, dim=1)
        h_prime = F.relu(torch.einsum('aij,ajk->aik',(attention, h)))
       
        coeff = torch.sigmoid(self.gate(torch.cat([x,h_prime], -1))).repeat(1,1,x.size(-1))
        retval = coeff*x+(1-coeff)*h_prime
        return retval
        
#multi_head gate module 
class MH_gate(torch.nn.Module):#用来做多头注意力代替原始实现以及加入不同的注意力偏执组合来看效果
    def __init__(self, n_in_feature, n_out_feature, attention_dropout_rate = 0.1, head_size = 8):
        super(MH_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature) 
        
        self.gate = nn.Linear(n_out_feature*2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.self_attention_norm = nn.LayerNorm(n_out_feature)
        # y = self.self_attention_norm(y)
        self.MH = MultiHeadAttention(n_out_feature, attention_dropout_rate =attention_dropout_rate , head_size =head_size )
        #head size must be equal hidden//attn_dim
        self.self_attention_dropout = nn.Dropout(attention_dropout_rate)
        # y = self.self_attention_dropout(y)

    def forward(self, x, adj,use_adj,attn_bias = None):#修改attn_bias 就可以达到添加偏置项的效果
        # print('x.shape ',x.shape)
        h = self.W(x)
        # print('h.shape ',h.shape)
        h = self.self_attention_norm(h)
        h = F.relu(self.MH(h,h,h,adj,use_adj,attn_bias))
        h_prime = self.self_attention_dropout(h)
        # print('h_prime.shape ',h_prime.shape)
        coeff = torch.sigmoid(self.gate(torch.cat([x,h_prime], -1))).repeat(1,1,x.size(-1))
        retval = coeff*x+(1-coeff)*h_prime#凸组合
        return retval
#transformer_gate
class Transformer_gate(nn.Module):
    def __init__(self, n_in_feature, n_out_feature,ffn_size, gate,dropout_rate = 0.1, attention_dropout_rate = 0.1, head_size = 8):
        super(Transformer_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature) 
        self.gate = nn.Linear(n_out_feature*2, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.gate_mode = gate
        self.self_attention_norm = nn.LayerNorm(n_out_feature)
        self.self_attention = MultiHeadAttention(n_out_feature, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(n_out_feature)
        self.ffn = FeedForwardNetwork(n_out_feature, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x,adj,use_adj,attn_bias=None):#不同的注意力的偏执测试
        y = self.W(x)
        y = self.self_attention_norm(y)
        y = self.self_attention(y, y, y, adj,use_adj,attn_bias)
        y = self.self_attention_dropout(y)
        h = x + y

        y = self.ffn_norm(h)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        h = h + y
        if self.gate_mode is True:
            coeff = torch.sigmoid(self.gate(torch.cat([x,h], -1))).repeat(1,1,x.size(-1))
            h = coeff*x+(1-coeff)*h#凸组合
            #gate or not

        return h
#pure transformer layer and graphformer layer! 这里输入需要时经过和GAT_gate 不同的编码层

class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.args = args
        self.self_attention_norm = GetGrpahNorm(self.args.n_out_feature,self.args.norm_type)
        self.self_attention = GetAttMode(self.args)
        self.self_attention_dropout = nn.Dropout(self.args.attention_dropout_rate)

        self.ffn_norm = GetGrpahNorm(self.args.n_out_feature,self.args.norm_type)

        self.ffn = FeedForwardNetwork(self.args.n_out_feature, self.args.ffn_size, self.args.dropout_rate)
        self.ffn_dropout = nn.Dropout(self.args.dropout_rate)

    def forward(self, x, adj,use_adj,attn_bias=None):#不同的注意力的偏执测试
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, adj,use_adj,attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x




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
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size
        assert hidden_size % head_size == 0
        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, adj,use_adj,attn_bias=None):
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
            # x = F.softmax(x, dim=-1)
            # x = x*
            #attention = F.dropout(attention, self.dropout, training=self.training)
            #h_prime = torch.matmul(attention, h)
            x = x*adj.unsqueeze(1)

        if attn_bias is not None:
            # print('MH ATT: ',x.shape,attn_bias.shape)
            x = x + attn_bias
        x = F.softmax(x, dim=-1)

        # x = torch.softmax(x, dim=-1)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x
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
        return DistangledMultiHeadAttention(args.n_out_feature, args.attention_dropout_rate, args.head_size)
    elif args.att_mode == 'SA':
        return MultiHeadAttention(args.n_out_feature, args.attention_dropout_rate, args.head_size)
    else:
        raise ValueError('not support this mode attention! until now! check your param')
def GetLayer(args):
    if args.fundation_model == 'paper':
        if args.layer_type == 'GAT_gate':
            return GAT_gate(args.n_in_feature,args.n_out_feature)
        if args.layer_type == 'MH_gate':
            
#参数里默认值0.1，8
            return MH_gate(args.n_in_feature, args.n_out_feature, args.attention_dropout_rate, args.head_size)
        if args.layer_type == 'Transformer_gate':
            return Transformer_gate(args.n_in_feature, args.n_out_feature,args.ffn_size,args.gate, args.dropout_rate, args.attention_dropout_rate, args.head_size)
    if args.fundation_model == 'graphformer':
        return EncoderLayer(args)
