import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import WeightAndSum
"""
with edge features
"""
from model.equiscore_layer import EquiScoreLayer
from utils.equiscore_utils import MLPReadout

class EquiScore(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        atom_dim = 16*12 if self.args.FP else 10*6
        self.atom_encoder = nn.Embedding(atom_dim  + 1, self.args.n_out_feature, padding_idx=0)
        self.edge_encoder = nn.Embedding( 36* 5 + 1, self.args.edge_dim, padding_idx=0) if args.edge_bias is True else nn.Identity()
        self.rel_pos_encoder = nn.Embedding(512, self.args.edge_dim, padding_idx=0) if args.rel_pos_bias is True else nn.Identity()#rel_pos
        self.in_degree_encoder = nn.Embedding(10, self.args.n_out_feature, padding_idx=0) if args.in_degree_bias is True else nn.Identity()

        if args.rel_pos_bias:
            self.linear_rel_pos =  nn.Linear(self.args.edge_dim, self.args.head_size) 
        if self.args.lap_pos_enc:
            self.embedding_lap_pos_enc = nn.Linear(self.args.pos_enc_dim, self.args.n_out_feature)
        self.layers = nn.ModuleList([ EquiScoreLayer(self.args) \
                for _ in range(self.args.n_graph_layer) ]) 

        self.MLP_layer = MLPReadout(self.args)   # 1 out dim if regression problem   
        self.weight_and_sum = WeightAndSum(self.args.n_out_feature)     
    def getAtt(self,g,full_g):
        h = g.ndata['x']

        h = self.atom_encoder(h.long()).mean(-2)

        if self.args.lap_pos_enc:
            h_lap_pos_enc = g.ndata['lap_pos_enc']
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.args.in_degree_bias:
            h = h+ self.in_degree_encoder(g.ndata['in_degree'])
        e = self.edge_encoder(g.edata['edge_attr']).mean(-2)
        for conv in self.layers:
            h, e = conv(g,full_g,h,e)
            h = F.dropout(h, p=self.args.dropout_rate, training=self.training)
            e = F.dropout(e, p=self.args.dropout_rate, training=self.training)
        # only ligand atom's features are used to task layer 
        h = h * g.ndata['V']
        hg = self.weight_and_sum(g,h)
        hg = self.MLP_layer(hg)
        
        return h,g,full_g,hg
    def getAttFirstLayer(self,g,full_g):
        """
        A tool function to get the attention of the first layer
        """
        h = g.ndata['x']

        h = self.atom_encoder(h.long()).mean(-2)

        if self.args.lap_pos_enc:
            h_lap_pos_enc = g.ndata['lap_pos_enc']
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.args.in_degree_bias:
            h = h+ self.in_degree_encoder(g.ndata['in_degree'])
        e = self.edge_encoder(g.edata['edge_attr']).mean(-2)

        for conv in [self.layers[0]]:
            h, e = conv(g,full_g,h,e)
            h = F.dropout(h, p=self.args.dropout_rate, training=self.training)
            e = F.dropout(e, p=self.args.dropout_rate, training=self.training)
        # only ligand atom's features are used to task layer
        h = h * g.ndata['V']
        hg = self.weight_and_sum(g,h)
        hg = self.MLP_layer(hg)
        
        return h,g,full_g,hg
    
    def forward(self, g, full_g):
        """
        Parameters
        ----------
        g : dgl.DGLGraph 
            convalent and IFP based graph 

        full_g :dgl.DGLGraph
            geometric based graph

		Returns
		-------
            probability of binding

        """
        h = g.ndata['x']
        h = self.atom_encoder(h.long()).mean(-2)

        if self.args.lap_pos_enc:
            h_lap_pos_enc = g.ndata['lap_pos_enc']
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.args.in_degree_bias:
            h = h+ self.in_degree_encoder(g.ndata['in_degree'])
        e = self.edge_encoder(g.edata['edge_attr']).mean(-2)

        for conv in self.layers:
            h, e = conv(g,full_g,h,e)
            h = F.dropout(h, p=self.args.dropout_rate, training=self.training)
            e = F.dropout(e, p=self.args.dropout_rate, training=self.training)
        h = h * g.ndata['V']
        hg = self.weight_and_sum(g,h)
        
        return self.MLP_layer(hg)
