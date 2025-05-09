from dataclasses import dataclass
import torch

import sys
import os

# Add the parent folder to sys.path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(parent_dir))

from models.ssl_models.custom_base import JointEmbeddingConfig, JointEmbeddingModel
from stable_ssl.utils import off_diagonal, gather_processes


class BarlowTwins(JointEmbeddingModel):
    """BarlowTwins model from [ZJM+21]_.

    Reference
    ---------
    .. [ZJM+21] Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021).
            Barlow Twins: Self-Supervised Learning via Redundancy Reduction.
            In International conference on machine learning (pp. 12310-12320). PMLR.
    """

    def initialize_modules(self):
        super().initialize_modules()
        self.bn = torch.nn.BatchNorm1d(self.config.model.projector[-1])

    @gather_processes
    def compute_ssl_loss(self, z_i, z_j):
        # Empirical cross-correlation matrix.
        c = self.bn(z_i).T @ self.bn(z_j)

        # Sum the cross-correlation matrix between all gpus.
        c.div_(self.config.data.train_dataset.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.config.model.lambd * off_diag
        return loss


@dataclass
class BarlowTwinsConfig(JointEmbeddingConfig):
    """Configuration for the BarlowTwins model parameters.

    Parameters
    ----------
    lambd : str
        Lambda parameter for the off-diagonal loss. Default is 0.1.
    """

    lambd: str = 0.1

    def trainer(self):
        return BarlowTwins
