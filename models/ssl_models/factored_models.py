import sys
import os

# Add the parent folder to sys.path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(parent_dir))

from dataclasses import dataclass
import torch

from stable_ssl.utils import off_diagonal, gather_processes
import logging
from dataclasses import dataclass, field
import torch

from stable_ssl.utils import mlp

from models.ssl_models.custom_base import JointEmbeddingConfig, JointEmbeddingModel
from models.ssl_models.custom_base import SelfDistillationModel, SelfDistillationConfig
import torch.nn.functional as F

from stable_ssl.utils import mlp, deactivate_requires_grad, update_momentum
from stable_ssl.base import BaseModel, ModelConfig
from models.ssl_models.create_nn import load_nn


# CovarianceFactorization method 1
class CovarianceFactorization(JointEmbeddingModel):
    """A modification on the BarlowTwins model from [ZJM+21]_.

    Reference
    ---------
    .. [ZJM+21] Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021).
            Barlow Twins: Self-Supervised Learning via Redundancy Reduction.
            In International conference on machine learning (pp. 12310-12320). PMLR.
    """

    def initialize_modules(self):
        self.bn = torch.nn.BatchNorm1d(self.config.model.projector[-1])
        backbone, fan_in = load_nn(
            backbone_model=self.config.model.backbone_model,
            pretrained=False,
            dataset=self.config.data.train_dataset.name,
        )
        self.backbone = backbone.train()

        sizes = [fan_in] + self.config.model.projector
        self.projector = mlp(sizes)

        self.backbone_classifier = torch.nn.Sequential(
            torch.nn.Linear(fan_in, self.config.data.train_dataset.num_classes),
            torch.nn.Sigmoid(),
        )
        self.projector_classifier = torch.nn.Linear(
            self.config.model.projector[-1],
            self.config.data.train_dataset.num_classes,
        )
        # the g_theta function in the architecture
        self.world_model = torch.nn.Sequential(
            torch.nn.Linear(self.config.model.projector[-1]+1,
                            256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.config.model.projector[-1]),
            torch.nn.Sigmoid(),
        )

    def compute_loss(self):
        embeddings = [self.backbone(view) for view in self.data[0][0]]
        # loss_backbone = self._compute_backbone_classifier_loss(*embeddings)

        projections = [self.projector(embed) for embed in embeddings]
        reconstruction = [self.world_model(torch.cat((projections[0], self.data[0][1]), dim=1))]  # append a_i to z_i and pass to world model
        # loss_proj = self._compute_projector_classifier_loss(*projections)
        loss_ssl = self.compute_ssl_loss(projections[0], projections[1], reconstruction[0])

        if self.global_step % self.config.log.log_every_step == 0:
            self.log(
                {
                    "train/loss_ssl": loss_ssl.item(),
                    # "train/loss_backbone_classifier": loss_backbone.item(),
                    # "train/loss_projector_classifier": loss_proj.item(),
                },
                commit=False,
            )

        return loss_ssl # + loss_proj + loss_backbone

    def forward(self, x):
        self.curr_actions = x[1]
        return self.backbone_classifier(self.backbone(x[0]))  # x is tuple of observation, action

    @gather_processes
    def compute_ssl_loss(self, z_i, z_j, z_j_reconstruction):
        # Empirical cross-correlation matrix.
        c = self.bn(z_i).T @ self.bn(z_j)

        # Sum the cross-correlation matrix between all gpus.
        c.div_(self.config.data.train_dataset.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.config.model.lambd * off_diag
        
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(z_j_reconstruction, z_j)
        loss += reconstruction_loss
        return loss


@dataclass
class CovarianceFactorizationConfig(JointEmbeddingConfig):
    """Configuration for the BarlowTwins model parameters.

    Parameters
    ----------
    lambd : str
        Lambda parameter for the off-diagonal loss. Default is 0.1.
    """

    lambd: str = 0.1

    def trainer(self):
        return CovarianceFactorization


# MaskingFactorizationConfig
"""MaskingFactorization model which is a modification of BYOL model."""
#
# Original Author: Hugues Van Assel
#         Randall Balestriero
#

class MaskingFactorization(SelfDistillationModel):
    """MaskingFactorization, a modification of BYOL model from [GSA+20].

    Reference
    ---------
    .. [GSA+20] Grill, J. B., Strub, F., Altché, ... & Valko, M. (2020).
            Bootstrap Your Own Latent-A New Approach To Self-Supervised Learning.
            Advances in neural information processing systems, 33, 21271-21284.
    """

    def initialize_modules(self):
        super().initialize_modules()

        sizes = [self.config.model.projector[-1]] + self.config.model.predictor
        self.predictor = mlp(sizes)

    def compute_ssl_loss(self, projections, projections_target):
        """Compute the loss of the BYOL model.

        Parameters
        ----------
        projections : list of torch.Tensor
            Projections of the different augmented views from the online network.
        projections_target : list of torch.Tensor
            Projections of the corresponding augmented views from the target network.

        Returns
        -------
        float
            The computed loss.
        """
        if len(projections) > 3 or len(projections_target) > 3:
            logging.warning(
                "MaskingFactorization only supports 3 views. Only the first two views will be used."
            )

        predictions = [self.predictor(proj) for proj in projections]

        sim = torch.nn.CosineSimilarity(dim=1)
        return -0.5 * (
            sim(predictions[0], projections_target[1]).mean()
            + sim(predictions[1], projections_target[0]).mean()
        )


@dataclass
class MaskingFactorizationConfig(SelfDistillationConfig):
    """Configuration for the MaskingFactorization model which is based on the BYOL model parameters.

    Parameters
    ----------
    predictor : str
        Architecture of the predictor head. Default is "2048-256".
    """

    predictor: list[int] = field(default_factory=lambda: [2048, 256])

    def __post_init__(self):
        """Convert predictor string to a list of integers if necessary."""
        super().__post_init__()
        if isinstance(self.predictor, str):
            self.predictor = [int(i) for i in self.predictor.split("-")]

    def trainer(self):
        return MaskingFactorization
