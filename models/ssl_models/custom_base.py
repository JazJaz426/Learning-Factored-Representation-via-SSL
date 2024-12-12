import copy
from abc import abstractmethod
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError

import logging
from disentanglement_metrics.disentangelement_callback import FrobeniusNorm, ZMinVar, MutualInformationGap

from stable_ssl.utils import load_nn, mlp, deactivate_requires_grad, update_momentum
from stable_ssl.base import BaseModel, ModelConfig


class JointEmbeddingModel(BaseModel):
    r"""Base class for training a joint-embedding SSL model.

    Parameters
    ----------
    config : GlobalConfig
        Parameters organized in groups.
        For details, see the `GlobalConfig` class in `config.py`.
    """

    def initialize_modules(self):
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
        self.curr_actions = None

    def forward(self, x):
        self.curr_actions = x[1]
        # logging.info(f"inside forward after indexing: {x[0].shape}")
        return self.backbone_classifier(self.backbone(x[0]))  # x is tuple of observation, action

    def compute_loss(self):
        # logging.info(self.data[0][0][0].shape)
        embeddings = [self.backbone(view) for view in self.data[0][0]]
        # loss_backbone = self._compute_backbone_classifier_loss(*embeddings)

        projections = [self.projector(embed) for embed in embeddings]
        # loss_proj = self._compute_projector_classifier_loss(*projections)
        loss_ssl = self.compute_ssl_loss(*projections)

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

    @abstractmethod
    def compute_ssl_loss(self, *projections):
        raise NotImplementedError

    def _compute_backbone_classifier_loss(self, *embeddings):
        print(self.data[1].shape, type(self.data[1]))
        print(self.data[1])
        losses = [
            F.cross_entropy(self.backbone_classifier(embed.detach()), self.data[1])
            for embed in embeddings
        ]
        return sum(losses)

    def _compute_projector_classifier_loss(self, *projections):
        losses = [
            F.cross_entropy(self.projector_classifier(proj.detach()), self.data[1])
            for proj in projections
        ]
        return sum(losses)

    def eval_step(self, name_loader):
        # logging.info(f"outside of the forward call: {self.data[0][0].shape}")
        output = self.forward(self.data[0])  # x is tuple of observations, action
        for name, metric in self.metrics.items():
            if name.startswith(f"eval/epoch/{name_loader}/"):
                metric.update(output, self.data[1])
            elif name.startswith(f"eval/step/{name_loader}/"):
                self.log({name: metric(output, self.data[1])}, commit=False)
        self.log(commit=True)

    def initialize_metrics(self):
        nc = self.config.data.datasets[self.config.data.train_on].num_classes

        # Initialize the metrics dictionary with the train metric.
        self.metrics = torch.nn.ModuleDict(
            {"train/step/mse": MeanSquaredError(num_outputs=nc)}
        )

        # Add unique evaluation metrics for each eval dataset.
        name_eval_loaders = set(self.dataloaders.keys()) - set(
            [self.config.data.train_on]
        )
        for name_loader in name_eval_loaders:
            self.metrics.update(
                {
                    f"eval/step/{name_loader}/mse": MeanSquaredError(num_outputs=nc),
                    f"eval/epoch/{name_loader}/mse": MeanSquaredError(num_outputs=nc),
                    f"eval/epoch/{name_loader}/forb_norm": FrobeniusNorm(),
                    # f"eval/epoch/{name_loader}/z_min_var": ZMinVar(),
                    # f"eval/epoch/{name_loader}/mig": MutualInformationGap(),
                }
            )


@dataclass
class JointEmbeddingConfig(ModelConfig):
    """Configuration for the joint-embedding model parameters.

    Parameters
    ----------
    projector : str
        Architecture of the projector head. Default is "2048-128".
    """

    projector: list[int] = field(default_factory=lambda: [2048, 128])

    def __post_init__(self):
        """Convert projector string to a list of integers if necessary."""
        if isinstance(self.projector, str):
            self.projector = [int(i) for i in self.projector.split("-")]


class SelfDistillationModel(JointEmbeddingModel):
    r"""Base class for training a self-distillation SSL model.

    Parameters
    ----------
    config : GlobalConfig
        Parameters organized in groups.
        For details, see the `GlobalConfig` class in `config.py`.
    """

    def initialize_modules(self):
        super().initialize_modules()
        self.backbone_target = copy.deepcopy(self.backbone)
        self.projector_target = copy.deepcopy(self.projector)

        deactivate_requires_grad(self.backbone_target)
        deactivate_requires_grad(self.projector_target)

    def compute_loss(self):
        embeddings = [self.backbone(view) for view in self.data[0][0]]  # x is tuple of observation, action
        # loss_backbone = self._compute_backbone_classifier_loss(*embeddings)

        projections = [self.projector(embed) for embed in embeddings]
        # loss_proj = self._compute_projector_classifier_loss(*projections)

        with torch.no_grad():
            projections_target = [
                self.projector_target(self.backbone_target(view))
                for view in self.data[0][0]  # x is tuple of observation, action
            ]
        loss_ssl = self.compute_ssl_loss(projections, projections_target)

        self.log(
            {
                "train/loss_ssl": loss_ssl.item(),
                # "train/loss_backbone_classifier": loss_backbone.item(),
                # "train/loss_projector_classifier": loss_proj.item(),
            },
            commit=False,
        )

        return loss_ssl # + loss_proj + loss_backbone

    def before_train_step(self):
        # Update the target parameters as EMA of the online model parameters.
        update_momentum(
            self.backbone, self.backbone_target, m=self.config.model.momentum
        )
        update_momentum(
            self.projector, self.projector_target, m=self.config.model.momentum
        )

    def eval_step(self, name_loader):
        # TODO update for triplet dataset
        # logging.info(f"outside of the forward call: {self.data[0][0].shape}")
        output = self.forward(self.data[0])  # x is tuple of observations, action
        for name, metric in self.metrics.items():
            if name.startswith(f"eval/epoch/{name_loader}/"):
                metric.update(output, self.data[1])
            elif name.startswith(f"eval/step/{name_loader}/"):
                self.log({name: metric(output, self.data[1])}, commit=False)
        self.log(commit=True)

    def initialize_metrics(self):
        nc = self.config.data.datasets[self.config.data.train_on].num_classes

        # Initialize the metrics dictionary with the train metric.
        self.metrics = torch.nn.ModuleDict(
            {"train/step/mse": MeanSquaredError(num_outputs=nc)}
        )

        # Add unique evaluation metrics for each eval dataset.
        name_eval_loaders = set(self.dataloaders.keys()) - set(
            [self.config.data.train_on]
        )
        for name_loader in name_eval_loaders:
            self.metrics.update(
                {
                    f"eval/step/{name_loader}/mse": MeanSquaredError(num_outputs=nc),
                    f"eval/epoch/{name_loader}/mse": MeanSquaredError(num_outputs=nc),
                    f"eval/epoch/{name_loader}/forb_norm": FrobeniusNorm(),
                    f"eval/epoch/{name_loader}/z_min_var": ZMinVar(),
                    f"eval/epoch/{name_loader}/mig": MutualInformationGap(),
                }
            )

@dataclass
class SelfDistillationConfig(JointEmbeddingConfig):
    """Configuration for the self-distillation model parameters.

    Parameters
    ----------
    momentum : float
        Momentum for the EMA of the target parameters. Default is 0.999.
    """

    momentum: float = 0.999
