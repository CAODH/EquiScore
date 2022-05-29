import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""

class MLPReadout(nn.Module):

    def __init__(self, args): #L=nb_hidden_layers
        super().__init__()
        self.args = args
        self.FC = nn.ModuleList([nn.Linear(self.args.n_out_feature, self.args.d_FC_layer) if i==0 else
                                nn.Linear(self.args.d_FC_layer, 2) if i==self.args.n_FC_layer-1  else
                                nn.Linear(self.args.d_FC_layer, self.args.d_FC_layer) for i in range(self.args.n_FC_layer)]) #4层 
        
    def forward(self, c_hs):
    
        for k in range(self.args.n_FC_layer):
          
            if k<self.args.n_FC_layer-1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.args.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.FC[k](c_hs)
        return c_hs