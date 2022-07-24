import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import dgl
from dgl.nn import WeightAndSum

"""
with edge features
    
"""
from GTE_layer import GTELayer
from mlp_readout_layer import MLPReadout

class GTENet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mu = nn.Parameter(torch.Tensor([args.initial_mu]).float())
        self.dev = nn.Parameter(torch.Tensor([args.initial_dev]).float())
        if args.auxiliary_loss:
            if args.deta_const:
                self.deta = 0.2
            else:
                self.deta = nn.Parameter(torch.Tensor([0.5]).float())

        atom_dim = 16*12 if self.args.FP else 10*6
        # self.in_feat_dropout = nn.Dropout(self.args.dropout)
        self.atom_encoder = nn.Embedding(atom_dim  + 1, self.args.n_out_feature, padding_idx=0)
        self.edge_encoder = nn.Embedding( 36* 5 + 1, self.args.edge_dim, padding_idx=0) if args.edge_bias is True else nn.Identity()
        self.rel_pos_encoder = nn.Embedding(512, self.args.edge_dim, padding_idx=0) if args.rel_pos_bias is True else nn.Identity()#rel_pos
        self.in_degree_encoder = nn.Embedding(10, self.args.n_out_feature, padding_idx=0) if args.in_degree_bias is True else nn.Identity()
        self.rel_3d_encoder = nn.Embedding(65, self.args.edge_dim, padding_idx=0) if args.rel_3d_pos_bias is True else nn.Identity()
        self.linear_3d_pos =  nn.Linear(self.args.edge_dim, self.args.head_size)
        if args.rel_pos_bias:
            self.linear_rel_pos =  nn.Linear(self.args.edge_dim, self.args.head_size) 
        if self.args.lap_pos_enc:
            self.embedding_lap_pos_enc = nn.Linear(self.args.pos_enc_dim, self.args.n_out_feature)
        self.layers = nn.ModuleList([ GTELayer(self.args) \
                for _ in range(self.args.n_graph_layer) ]) 
        # self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(self.args)   # 1 out dim since regression problem   
        self.weight_and_sum = WeightAndSum(self.args.n_out_feature)     
    def getAtt(self,g,full_g):
        h = g.ndata['x']
        # print('max,min atom fea',torch.max(h),torch.min(h))
        h = self.atom_encoder(h.long()).mean(-2)
        # h = self.in_feat_dropout(h)
        if self.args.lap_pos_enc:
            h_lap_pos_enc = g.ndata['lap_pos_enc']
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.args.in_degree_bias:
            h = h+ self.in_degree_encoder(g.ndata['in_degree'])
        e = self.edge_encoder(g.edata['edge_attr']).mean(-2)
        if self.args.only_dis_adj2:

            full_g.edata['adj2'] = torch.where(full_g.edata['adj2'] > self.mu,torch.exp(-torch.pow(full_g.edata['adj2']-self.mu, 2)/(self.dev + 1e-6)),\
                torch.tensor(1.0).to(self.dev.device))+ full_g.edata['adj1']
        if self.args.rel_3d_pos_bias:
            rel_3d_bias = self.rel_3d_encoder(full_g.edata['rel_pos_3d'])#.permute(0, 3, 1, 2)
            rel_3d_bias = nn.functional.relu(rel_3d_bias)
            full_g.edata['rel_pos_3d'] = self.linear_3d_pos(rel_3d_bias).view(-1,self.args.head_size).contiguous().float()
        # convnets
        for conv in self.layers:
            h, e = conv(g,full_g,h,e)
            h = F.dropout(h, p=self.args.dropout_rate, training=self.training)
            e = F.dropout(e, p=self.args.dropout_rate, training=self.training)
        return g,full_g

    def forward(self, g, full_g):
        h = g.ndata['x']
        # print('max,min atom fea',torch.max(h),torch.min(h))
        h = self.atom_encoder(h.long()).mean(-2)
        # h = self.in_feat_dropout(h)
        if self.args.lap_pos_enc:
            h_lap_pos_enc = g.ndata['lap_pos_enc']
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.args.in_degree_bias:
            h = h+ self.in_degree_encoder(g.ndata['in_degree'])
        e = self.edge_encoder(g.edata['edge_attr']).mean(-2)
        if self.args.only_dis_adj2:

            full_g.edata['adj2'] = torch.where(full_g.edata['adj2'] > self.mu,torch.exp(-torch.pow(full_g.edata['adj2']-self.mu, 2)/(self.dev + 1e-6)),\
                torch.tensor(1.0).to(self.dev.device))+ full_g.edata['adj1']
        if self.args.rel_3d_pos_bias:
            rel_3d_bias = self.rel_3d_encoder(full_g.edata['rel_pos_3d'])#.permute(0, 3, 1, 2)
            rel_3d_bias = nn.functional.relu(rel_3d_bias)
            full_g.edata['rel_pos_3d'] = self.linear_3d_pos(rel_3d_bias).view(-1,self.args.head_size).contiguous().float()
        # convnets
        for conv in self.layers:
            h, e = conv(g,full_g,h,e)
            h = F.dropout(h, p=self.args.dropout_rate, training=self.training)
            e = F.dropout(e, p=self.args.dropout_rate, training=self.training)
        # select ligand atom for predict
        # g.ndata['x'] = h * g.ndata['V']
        # hg = dgl.sum_nodes(g, 'x')#/dgl.sum_nodes(g,'V') # mean add sum or max or min later! concat
        hg = self.weight_and_sum(g,h)
        return self.MLP_layer(hg)
