import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import numpy as np
import os
# os.path.append('../utils')
from utils.equiscore_utils import *
"""
    Multi Attention Head
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
            nn.Linear(edge_dim, num_heads))
        
    def propagate_attention(self, g,full_g):
        """
        attention propagation with proir informations
        Parameters
        ----------
        g : dgl.DGLGraph 
            convalent and IFP based graph 

        full_g :dgl.DGLGraph
            geometric based graph
	
		Returns
		-------
        
        """

        ############### geometric distance based graph attention module ################################
        full_g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))
        ################################## transform coors as rel distance to decay attention score ####
        full_g.apply_edges(fn.u_sub_v('coors', 'coors', 'detla_coors')) 
        full_g.apply_edges(square('detla_coors', 'rel_pos_3d'))

        full_g.edata['rel_pos_3d'] = self.coors_mlp(full_g.edata['rel_pos_3d'].float())
        # scaling
        full_g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        ########################################
        # distance gate 
        full_g.apply_edges(guss_decoy('score','rel_pos_3d'))
        # full_g.edata['score'] = full_g.edata['score'].sum(-1, keepdim=True) # only be used to ablation study
        ##########################################
        # read score on structual based edges
        # sent attn score to structual based edges
        src,dst = g.edges() # get structual based edges
        g.edata['score'] = full_g.edge_subgraph(full_g.edge_ids(src,dst),relabel_nodes=False).edata['score']
        # project score to edge features to update features
        g.edata['e_out'] = self.attn_proj(g.edata['score'].view(-1,self.num_heads).contiguous()) # score to edge features
        ############### structual edges(covalent bond based edges) bias################################
        # Compute attention score bias
        g.apply_edges(edge_bias('score', 'proj_e'))  # add edge bias 
        # add edge_bias and update on geometric distanced based edges
        full_g.apply_edges(func=partUpdataScore('score','score',g),edges=g.edges()) # 
        # Copy edge features as e_out to be passed to FFN_e
        ###################################################
        # softmax
        # for softmax numerical stability
        eids = full_g.edges()
        ################################
        full_g.edata['score'] = edge_softmax(graph = full_g,logits = full_g.edata['score'].clamp(-5,5))
        ############## score as coors update factor and update vector features ##############
        full_g.apply_edges(edge_mul_score('detla_coors', 'score'))# accumlate detla_coors 
        full_g.send_and_recv(eids, dgl.function.copy_e('detla_coors','detla_coors'), fn.sum('detla_coors', 'coors_add'))
        # to update vector features
        full_g.ndata['coors'] += full_g.ndata['coors_add'] 
        #################################################################
        #########################################################
        # attention dropout for control overfitting
        full_g.edata['score'] = self.attn_dropout(full_g.edata['score'])
        #feature update
        full_g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))



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
        ########################## norm coors for EquiScore ############### 
        full_g.ndata['coors'] = self.coor_norm(full_g.ndata['coors'])

        self.propagate_attention(g,full_g)
        e_out = self.output_layer_edge(g.edata['e_out'] + e)
        h_out = full_g.ndata['wV'] 
        h_out = self.output_layer(h_out.view(-1, self.out_dim * self.num_heads))
        return h_out, e_out
    
class EquiScoreLayer(nn.Module):
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

        self.FFN_e_layer = FeedForwardNetwork(self.args.edge_dim, self.args.ffn_size, self.args.dropout_rate)
 
        self.layer_norm2_h = GraphNorm(hidden_dim = self.args.n_out_feature)
        self.layer_norm2_e = nn.LayerNorm(self.args.edge_dim)
            
    def forward(self, g, full_g,x, e):
        """
        update the node embedding and edge embedding
        Parameters
        ----------
        g : dgl.DGLGraph 
            convalent and IFP based graph 

        full_g :dgl.DGLGraph
            geometric based graph
        x : torch.Tensor
            nodes embeddings
        e : torch.Tensor
            edges embeddings
	
		Returns
		-------
        x : torch.Tensor
            updated nodes embeddings
        e : torch.Tensor
            updated edges embeddings
        
        """

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