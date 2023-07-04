import torch.nn as nn
import torch
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""
class MLPReadout(nn.Module):

    def __init__(self, args): 
        super().__init__()
        self.args = args
        self.FC = nn.ModuleList([nn.Linear(self.args.n_out_feature, self.args.d_FC_layer) if i==0 else
                                nn.Linear(self.args.d_FC_layer, 2) if i==self.args.n_FC_layer-1  else
                                nn.Linear(self.args.d_FC_layer, self.args.d_FC_layer) for i in range(self.args.n_FC_layer)])
        
    def forward(self, c_hs):
    
        for k in range(self.args.n_FC_layer):
          
            if k<self.args.n_FC_layer-1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.args.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.FC[k](c_hs)
        return c_hs
class CoorsNorm(nn.Module):
    """
    Norm the coors

    """
    def __init__(self, eps = 1e-8, scale_init = 1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        return normed_coors * self.scale

class GraphNorm(nn.Module):

    def __init__(self, norm_type = 'gn', hidden_dim=64, print_info=None):
        super(GraphNorm, self).__init__()
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

    def forward(self, graph, tensor, print_=False):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes()
        batch_size = len(batch_list)
        batch_list = batch_list.long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias

class GraphNorm_no_mean_scale(nn.Module):
    """
    GraphNorm without mean scale
    """
    def __init__(self, num_features, eps=1e-5, affine=True, is_node=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        self.is_node = is_node
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def norm(self, x):
        mean = x.mean(dim = 0, keepdim = True)
        var = x.std(dim = 0, keepdim = True)
        x = (x - mean) / (var + self.eps)
        return x

    def forward(self, g, x):
        graph_size  = g.batch_num_nodes() if self.is_node else g.batch_num_edges()
        # print(graph_size)
        x_list = torch.split(x, tuple(graph_size.cpu().numpy()))
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = torch.cat(norm_list, 0)

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x
def fourier_encode_dist(x, num_encodings = 4, include_self = True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x
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
       edge_bias : edge bias from edge features
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
def guss_decoy(field,rel_pos):
    """
    rel_pos :3d distance pass a linear layer or FFN
    """
    def func(edges):
        
        return {field: edges.data[field].sum(-1, keepdim=True)*edges.data[rel_pos].unsqueeze(-1)}
    
    return func

def partUpdataScore(out_filed,in_filed,graph_sparse):
    """update part of geometric distance based graph edge score"""
    def func(edges):
        return {out_filed:graph_sparse.edata[in_filed]}
    return func