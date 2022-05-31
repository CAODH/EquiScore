import torch.nn as nn
import torch
import torch
import torch.nn as nn

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
        Param: []
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