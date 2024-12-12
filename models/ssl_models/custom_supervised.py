import copy
from abc import abstractmethod
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError

import logging
from disentanglement_metrics.disentangelement_callback import FrobeniusNorm, ZMinVar, MutualInformationGap

from stable_ssl.utils import load_nn
from stable_ssl.base import BaseModel, ModelConfig


class Supervised(BaseModel):
    r"""Base class for training a supervised model.

    Parameters
    ----------
    config : BaseModelConfig
        Parameters for BaseModel organized in groups.
        For details, see the `BaseModelConfig` class in `config.py`.
    """

    def initialize_modules(self):
        backbone, fan_in = load_nn(
            backbone_model=self.config.model.backbone_model,
            pretrained=False,
            dataset=self.config.data.train_dataset.name,
        )
        self.backbone = backbone.train()
        
        self.backbone_classifier = torch.nn.Sequential(
            torch.nn.Linear(fan_in, self.config.data.train_dataset.num_classes),
            torch.nn.Sigmoid(),
        )
        self.curr_actions = None

    def forward(self, x):
        self.curr_actions = x[1]
        # logging.info(f"inside forward after indexing: {x[0].shape}")
        return self.backbone_classifier(self.backbone(x[0]))  # x is tuple of observation, action

    def compute_loss(self):
        # data is (x,y) so data[0] will be x since tuple indexing. Now data[0] is (observation, action)
        predictions = self.forward(self.data[0])
        loss = F.mse_loss(predictions, self.data[1])

        if self.global_step % self.config.log.log_every_step == 0:
            self.log(
                {
                    "train/loss": loss.item(),
                },
                commit=False,
            )

        return loss

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
class SupervisedConfig(ModelConfig):
    """
    Configuration for the supervised model.
    """
    
    def trainer(self):
        return Supervised
