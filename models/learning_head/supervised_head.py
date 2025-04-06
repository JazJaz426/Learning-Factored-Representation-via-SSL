import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from typing import Optional, List

class SupervisedLearner(nn.Module):

    def __init__(self, backbone_dim: int, vector_size_per_factor:int, num_factors:List[int]):
        super(SupervisedLearner, self).__init__()

        self.expert_scale_proj = nn.Sequential(
                                nn.Linear(backbone_dim, len(num_factors)*vector_size_per_factor),
                                nn.ReLU(),
                                nn.Linear(len(num_factors)*vector_size_per_factor, len(num_factors)*vector_size_per_factor)
        )
        
        self.expert_distribution_proj = nn.ModuleList([nn.Linear(vector_size_per_factor, out_dim+1) for out_dim in num_factors])
        self.softmax = nn.Softmax(dim=-1)

        #extra variables to reshape output
        self.num_dims = len(num_factors)
        self.vector_size_per_factor = vector_size_per_factor
    
    def forward(self, x: torch.Tensor, test:bool=True)->torch.Tensor:
        
        #if in eval mode: used for PPO policy learning => then detach computation and take argmax
        if test:
            x = self.expert_scale_proj(x)
            x = x.reshape(x.shape[0], self.num_dims, self.vector_size_per_factor)
            x = [torch.argmax(self.softmax(fc_proj(x[:,i,:])).detach(), dim=-1) for i, fc_proj in enumerate(self.expert_distribution_proj)] #list of length expert_dim: every entry is (batch_size) tensor
            x = torch.stack(x, dim=0).T #NOTE: hopefully a (batch size, expert_dim) tensor
            x = x.float() #convert to float for loss and forward pass
        #if in train mode: used for SL representation learning  ==> then do not detach computation and do not take argmax
        else:
            x = self.expert_scale_proj(x)
            x = x.reshape(x.shape[0], self.num_dims, self.vector_size_per_factor)
            x = [self.softmax(fc_proj(x[:,i,:])) for i, fc_proj in enumerate(self.expert_distribution_proj)]
            # x = torch.stack(x, dim=0).T #NOTE: hopefully a (batch size, expert_dim, num options per feature)
        
        return x




