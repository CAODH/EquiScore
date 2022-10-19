import torch
import torch.nn as nn
import torch.nn.functional as F
from graphnorm import GraphNorm
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import time
import numpy as np
from LGEG_utils import *
"""
     with edge features
"""
"""
    Util functions
"""
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func
def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

def edge_bias(node_attn, bias_edge):
    """
       node attn score :node_attn
       edge_bias : edge bias from edge feas
    """
    def func(edges):
        return {node_attn: (edges.data[node_attn] + edges.data[bias_edge])}   
    return func
# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat] }
    return func
def edge_mul_score(field,score):
    def func(edges):
        # print(edges.data[field].shape,edges.data[score].shape )
        return {field:edges.data[field]*edges.data[score].sum(1).reshape(-1,1)}
    return func
def exp(field):
    def func(edges):
         return {field: torch.exp((edges.data[field]))}
    return func
def square(field,out_field):
    def func(edges):
         return {out_field: torch.square((edges.data[field])).sum(dim = -1,keepdim = True)}
    return func
# for global decoy func
def guss_decoy(field,rel_pos):
    '''
    adj ; 3d distance with decay
    like:
        full_g.edata['adj2'] = torch.where(full_g.edata['adj2'] > self.mu,torch.exp(-torch.pow(full_g.edata['adj2']-self.mu, 2)/(self.dev + 1e-6)),\
                    torch.tensor(1.0).to(self.dev.device))+ full_g.edata['adj1']
    rel_pos :3d distance pass a linear
    '''
    def func(edges):
        # print()
        # return {field: edges.data[field].sum(-1, keepdim=True)*edges.data[adj].unsqueeze(1) + edges.data[rel_pos].unsqueeze(-1)}
        return {field: edges.data[field].sum(-1, keepdim=True)*edges.data[rel_pos].unsqueeze(-1)}
        # return {field: edges.data[field].sum(-1, keepdim=True)*edges.data[adj].unsqueeze(1) + edges.data[rel_pos].unsqueeze(-1)}
       
    return func

def partUpdataScore(out_filed,in_filed,graph_sparse):
    '''update part of full graph edge score'''
    def func(edges):
        return {out_filed:graph_sparse.edata[in_filed]}
    return func

"""
    Single Attention Head
"""
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(ffn_size, hidden_size)
    def forward(self, x):
        x = self.ffn_dropout(self.layer1(x))
        x = self.gelu(x)
        x = self.layer2(x)
        return x
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads,edge_dim,dropout_rate = 0.2):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.Q = nn.Linear(in_dim, self.out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, self.out_dim * num_heads, bias=True)
        self.V = nn.Linear(in_dim, self.out_dim * num_heads, bias=True)
        self.proj_e = nn.Linear(edge_dim, num_heads, bias=True)
        self.attn_proj = nn.Linear(num_heads,edge_dim)
        self.output_layer = nn.Linear(self.out_dim * num_heads, self.out_dim * num_heads)
        self.output_layer_edge = nn.Linear(edge_dim, edge_dim)
        self.coor_norm = CoorsNorm()
        self.coors_mlp = nn.Sequential(
            nn.Linear(1, edge_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(edge_dim, num_heads))#.view(-1,self.args.head_size).contiguous().float()
        
    def propagate_attention(self, g,full_g):

        ############### global module start ################################
        # apply sparse graph nodes fea to dense graph for global attention module
        full_g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        ################################## transform coors as rel distance to decay attention score #########################################
        full_g.apply_edges(fn.u_sub_v('coors', 'coors', 'detla_coors')) #, edges)
        full_g.apply_edges(square('detla_coors', 'rel_pos_3d')) #, edges)
        # print(full_g.edata['rel_pos_3d'].shape,full_g.edata['rel_pos_3d'][:20])
        full_g.edata['rel_pos_3d'] = self.coors_mlp(full_g.edata['rel_pos_3d'].float())
        # scaling
        # rel_dist decay 

        full_g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        ########################################
        # distance decay
        full_g.apply_edges(guss_decoy('score','rel_pos_3d'))# distance decay ,best model need this update!! 
        # full_g.edata['score'] = full_g.edata['score'].sum(-1, keepdim=True)# only be  used to ablation study

        ##########################################
        # update score from edge 
        #  sent attn score to sparse edges
        src,dst = g.edges() # get sparse edges
        g.edata['score'] = full_g.edge_subgraph(full_g.edge_ids(src,dst),relabel_nodes=False).edata['score']
        g.edata['e_out'] = self.attn_proj(g.edata['score'].view(-1,self.num_heads).contiguous()) # score to edge feas
        ############### local module start ################################
        # Compute attention score
        g.apply_edges(edge_bias('score', 'proj_e'))  # add edge bias 
        # add edge_bias to full_g 
        full_g.apply_edges(func=partUpdataScore('score','score',g),edges=g.edges()) # best model need this module ,ablation to # it only!!!
        # Copy edge features as e_out to be passed to FFN_e
        ###################################################


        # softmax
        # for softmax numerical stability
        eids = full_g.edges()
        ################################
        full_g.edata['score'] = edge_softmax(graph = full_g,logits = full_g.edata['score'].clamp(-5,5))
        ############## score as coors update factor and update coors ##############
        ##
        full_g.apply_edges(edge_mul_score('detla_coors', 'score'))# accu detla_coors 
        full_g.send_and_recv(eids, dgl.function.copy_e('detla_coors','detla_coors'), fn.sum('detla_coors', 'coors_add'))
        full_g.ndata['coors'] += full_g.ndata['coors_add']# BEST MODEL IS full_g.ndata['coors'] += full_g.ndata['coors_add']
        #################################################################

        #########################################################

        full_g.edata['score'] = self.attn_dropout(full_g.edata['score'])
        full_g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        # full_g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z')) # div
        ############### global module end ################################
        ############## coors update factor##############

    def forward(self, g, full_g,h, e):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)

        # get projections for multi-head attention
        full_g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        full_g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        full_g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, 1)
        # g.edata['attn_proj'] = 
        ########################## norm coors for LGEG model############### 
        full_g.ndata['coors'] = self.coor_norm(full_g.ndata['coors'])

        self.propagate_attention(g,full_g)
        e_out = self.output_layer_edge(g.edata['e_out'] + e)
        h_out = full_g.ndata['wV'] 
        h_out = self.output_layer(h_out.view(-1, self.out_dim * self.num_heads))
        return h_out, e_out
    
class GTELayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.attention = MultiHeadAttentionLayer(self.args.n_out_feature, self.args.n_out_feature//self.args.head_size, self.args.head_size,self.args.edge_dim,self.args.attention_dropout_rate)
        self.self_ffn_dropout = nn.Dropout(self.args.dropout_rate)
        self.self_ffn_dropout_2 = nn.Dropout(self.args.dropout_rate)
        self.ffn_dropout_edge = nn.Dropout(self.args.dropout_rate)
        self.ffn_dropout_edge_2 = nn.Dropout(self.args.dropout_rate)
        self.layer_norm1_h = GraphNorm(hidden_dim = self.args.n_out_feature)
        self.layer_norm1_e = nn.LayerNorm(self.args.edge_dim)
        # FFN for h
        self.FFN_h_layer = FeedForwardNetwork(self.args.n_out_feature, self.args.ffn_size, self.args.dropout_rate)
        # self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        # FFN for e
        self.FFN_e_layer = FeedForwardNetwork(self.args.edge_dim, self.args.ffn_size, self.args.dropout_rate)
        # self.FFN_e_layer2 = nn.Linear(self.args.edge_dim*2, self.args.edge_dim)
        self.layer_norm2_h = GraphNorm(hidden_dim = self.args.n_out_feature)
        self.layer_norm2_e = nn.LayerNorm(self.args.edge_dim)
            

        
    def forward(self, g, full_g,x, e):
        y = self.layer_norm1_h(g,x)
        e_norm = self.layer_norm1_e(e)
        y, e_norm = self.attention(g,full_g, y, e_norm)
        e_norm = self.ffn_dropout_edge(e_norm)
        e = e + e_norm
        e_norm = self.layer_norm2_e(e)
        e_norm = self.FFN_e_layer(e_norm)
        e_norm = self.ffn_dropout_edge_2(e_norm)
        e = e + e_norm
        # x layer module
        y = self.self_ffn_dropout(y)
        x = x + y
        y =  self.layer_norm2_h(g,x)
        y =  self.FFN_h_layer(y)
        y = self.self_ffn_dropout_2(y)
        x = x + y
        return x, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={})'.format(self.__class__.__name__,
                                             self.args.n_out_feature,
                                             self.args.n_out_feature, self.head_size)