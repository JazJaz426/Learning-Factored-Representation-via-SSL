import sys
import os

# Add the parent folder to sys.path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(parent_dir))

from .frob_norm_diff_identity import frob_norm_diff_identity
from .z_min_var import z_min_var
from .mutual_info import mig
import torch
from torchmetrics import Metric

class FrobeniusNorm(Metric):
    def __init__(self):
        super().__init__()
        # Initialize metric state variables
        self.add_state("total_frob_norm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
       

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        '''
        Inputs:
            preds: factored representation encoder's output is predicted
            target: ground truth expert feature representations
        '''

        #compute the Frobenius Norm metric 
        frob_norm = frob_norm_diff_identity(preds.detach().cpu().numpy())

        #increment total Frobenius norm and total to get avg frobenius norm across batch/count
        self.total_frob_norm += frob_norm
        self.total += target.size()[0]

    def compute(self):
        # Compute final metric
        return self.total_frob_norm.float() / self.total

class ZMinVar(Metric):
    def __init__(self):
        super().__init__()
        # Initialize metric state variables
        self.add_state("total_z_min_var", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
       

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        '''
        Inputs:
            preds: factored representation encoder's output is predicted
            target: ground truth expert feature representations
        '''

        #compute the Frobenius Norm metric 
        train_size = int(0.8*target.shape[-1])
        test_size = target.shape[-1] - train_size
        var_val = z_min_var(
            factors=target.detach().cpu().numpy(),
            codes=preds.detach().cpu().numpy(),
            nb_training=train_size,
            nb_eval=test_size
        )

        #increment total Frobenius norm and total to get avg frobenius norm across batch/count
        self.total_z_min_var += var_val
        self.total += target.numel()

    def compute(self):
        # Compute final metric
        return self.total_z_min_var.float() / self.total



class MutualInformationGap(Metric):
    def __init__(self):
        super().__init__()
        # Initialize metric state variables
        self.add_state("total_mig", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
       

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        '''
        Inputs:
            preds: factored representation encoder's output is predicted
            target: ground truth expert feature representations
        '''

        #compute the Frobenius Norm metric 
        mig_val = mig(factors= target.detach().cpu().numpy(), codes= preds.detach().cpu().numpy())

        #increment total Frobenius norm and total to get avg frobenius norm across batch/count
        self.total_mig += mig_val
        self.total += target.numel()

    def compute(self):
        # Compute final metric
        return self.total_mig.float() / self.total