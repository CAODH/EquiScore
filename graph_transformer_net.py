import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    Graph Transformer with edge features
    
"""
from graph_transformer_edge_layer import GraphTransformerLayer
from mlp_readout_layer import MLPReadout

class GraphTransformerNet(nn.Module):
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

        atom_dim = 16*10 if self.args.FP else 10*5
        self.in_feat_dropout = nn.Dropout(self.args.dropout)
        self.atom_encoder = nn.Embedding(atom_dim  + 1, self.args.n_out_feature, padding_idx=0)
        self.edge_encoder = nn.Embedding( 35* 5 + 1, self.args.edge_dim, padding_idx=0) if args.edge_bias is True else nn.Identity()
        self.rel_pos_encoder = nn.Embedding(512, self.args.head_size, padding_idx=0) if args.rel_pos_bias is True else nn.Identity()#rel_pos
        self.in_degree_encoder = nn.Embedding(10, self.args.n_out_feature, padding_idx=0) if args.in_degree_bias is True else nn.Identity()
        self.rel_3d_encoder = nn.Embedding(65, self.args.edge_dim, padding_idx=0) if args.rel_3d_pos_bias is True else nn.Identity()
        self.linear_3d_pos =  nn.Linear(self.args.edge_dim, self.args.head_size,bias = False)
        if self.args.lap_pos_enc:
            self.embedding_lap_pos_enc = nn.Linear(self.args.pos_enc_dim, self.args.n_out_feature)
        self.layers = nn.ModuleList([ GraphTransformerLayer(self.args.n_out_feature,self.args.n_out_feature, \
            self.args.head_size, self.args.dropout,self.args.layer_norm, self.args.batch_norm, self.args.residual) \
                for _ in range(self.args.n_graph_layer) ]) 
        # self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(self.args.n_out_feature, 2)   # 1 out dim since regression problem        
        
    def forward(self, g, full_g):

        # input embedding
        h = g.ndata['x']

        h = self.atom_encoder(h.long()).mean(-2)
        h = self.in_feat_dropout(h)
        if self.args.lap_pos_enc:
            h_lap_pos_enc = g.ndata['pos_lp_enc']
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.args.in_degree_bias:
            h = h+ self.in_degree_encoder(g.ndata['in_degree'])
        e = self.edge_encoder(g.edata['edge_attr']).mean(-2)
        # full_g and rel_pos or 3d pos 
        if self.args.rel_pos_bias:

            full_g['rel_pos_bias'] = None
        if self.args.args.rel_3d_pos_bias:
            full_g['args.rel_3d_pos_bias'] = None
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        # select ligand atom for predict
        g.ndata['h'] = h * g.ndata['V']

        hg = dgl.sum_nodes(g, 'h')/torch.sum(g.ndata['V'])
        
            
        return self.MLP_layer(hg)
