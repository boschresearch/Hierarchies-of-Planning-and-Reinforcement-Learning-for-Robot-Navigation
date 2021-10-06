# Hierarchies of Planning and Reinforcement Learning for Robot Navigation
# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This source code is derived from PyTorch-RL
#   (https://github.com/Khrylx/PyTorch-RL)
# Copyright (c) 2020 Ye Yuan, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree,
# which derived it from pytorch-trpo
#   (https://github.com/ikostrikov/pytorch-trpo)
# Copyright (c) 2017 Ilya Kostrikov, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import math
import numpy as np
from typing import Tuple, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lib.policy.policy import PGSupportingPolicy


class ContinuousMLP(PGSupportingPolicy):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 is_disc_action: bool,
                 hidden_sizes: Optional[Tuple[int]] = None,
                 embedding: Optional[Callable] = None,
                 activation: Callable = F.elu,
                 log_std: int = 0,
                 **kwargs) -> None:
        """
        :param obs_dim: int, size of the flat observation space
        :param action_dim: int, size of the flat action space
        :param is_disc_action: bool, specifies whether the policy's output is a discrete action.
                               Is required to be False.
        :param hidden_sizes: Optional[Tuple[int]], size of the hidden layers.
                             Default is (64, 64, 64)
        :param embedding: Embedding module of type nn.Module
        :param activation: PyTorch functional activation (e.g. F.elu)
        :param log_std: initial value of log std
        :param kwargs: Dict, key-word arguments to parent's constructor
        """
        if is_disc_action:
            raise ValueError('Class ContinuousMLP does not support discrete action spaces.')

        if hidden_sizes is None:
            hidden_sizes = (64, 64, 64)

        super().__init__(obs_dim, action_dim, is_disc_action, **kwargs)
        self.embedding = embedding

        self.activation = activation

        self.affine_layers = nn.ModuleList()
        last_dim = obs_dim
        for hidden_size in hidden_sizes:
            self.affine_layers.append(nn.Linear(last_dim, hidden_size))
            last_dim = hidden_size

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, x: Tensor, embedding_features: Optional[Tensor] = None) \
            -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        :param x: State of Shape Batch x Features
        :param embedding_features: Either None or Batch x Sequence x Features
        :return: Mean, log of standard deviation and standard deviation of
        action distribution and additional embedding information if available
        """

        embedding_info = None
        if self.embedding is not None:
            assert embedding_features is not None, \
                "Can't run forward pass: Missing additional_features"
            embedding_output = self.embedding(embedding_features)

            # Some feature representations return additional information
            # about the embeddings (e.g. PointNet)
            if type(embedding_output) == tuple:
                x = torch.cat((x, embedding_output[0]), 1)
                embedding_info = embedding_output[1]
            else:
                x = torch.cat((x, embedding_output), 1)

        for affine in self.affine_layers:
            x = self.activation(affine(x))
        action_mean = self.action_mean(x)

        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std, embedding_info

    def select_action(self, x: Tensor, squash=False) -> float:
        action_mean, _, action_std, _ = self.forward(x)
        action = torch.normal(action_mean, action_std)
        if squash:
            action = 30 * torch.tanh(action)
        return action.detach().cpu().numpy()[0]

    def get_log_prob(self, x: Tensor, actions: Tensor) -> Tensor:
        action_mean, action_log_std, action_std, _ = self.forward(x)
        return self._normal_log_density(
            actions, action_mean, action_log_std, action_std)

    def get_kl(self, x: torch.Tensor):
        mean1, log_std1, std1, _ = self.forward(x)
        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    @staticmethod
    def _normal_log_density(x, mean, log_std, std):
        var = std.pow(2)
        log_density = -(x - mean).pow(2) / (2 * var) - \
            0.5 * math.log(2 * math.pi) - log_std
        return log_density.sum(1, keepdim=True)


class ContinuousMLPDual(PGSupportingPolicy):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 is_disc_action: bool,
                 hidden_sizes: Optional[Tuple[int]] = None,
                 embedding: Optional[Callable] = None,
                 activation: Callable = F.elu,
                 log_std: int = 0,
                 **kwargs) -> None:
        """
        :param obs_dim: int, size of the flat observation space
        :param action_dim: int, size of the flat action space
        :param is_disc_action: bool, specifies whether the policy's output is a discrete action.
                               Is required to be False.
        :param hidden_sizes: Optional[Tuple[int]], size of the hidden layers.
                             Default is (64, 64, 64)
        :param embedding: Embedding module of type nn.Module
        :param activation: PyTorch functional activation (e.g. F.elu)
        :param log_std: initial value of log std
        :param kwargs: Dict, key-word arguments to parent's constructor
        """
        if is_disc_action:
            raise ValueError('Class ContinuousMLP does not support discrete action spaces.')

        if hidden_sizes is None:
            hidden_sizes = (64, 64, 64)

        super().__init__(obs_dim, action_dim, is_disc_action, **kwargs)
        self.embedding = embedding

        self.activation = activation

        self.affine_layers = nn.ModuleList()
        last_dim = obs_dim
        for hidden_size in hidden_sizes:
            self.affine_layers.append(nn.Linear(last_dim, hidden_size))
            last_dim = hidden_size

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

        self.vf_head = nn.Linear(last_dim, 1)

    def forward(self, x):

        for affine in self.affine_layers:
            x = self.activation(affine(x))
        action_mean = self.action_mean(x)

        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        value_output = self.vf_head(x)

        return action_mean, action_log_std, action_std, value_output

    def get_value(self, x):
        _, _, _, val = self.forward(x)
        return val

    def select_action(self, x: Tensor) -> float:
        action_mean, _, action_std, _ = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action.detach().cpu().numpy()[0]

    def get_log_prob(self, x: Tensor, actions: Tensor) -> Tensor:
        action_mean, action_log_std, action_std, _ = self.forward(x)
        return self._normal_log_density(
            actions, action_mean, action_log_std, action_std)

    def get_kl(self, x: torch.Tensor):
        mean1, log_std1, std1, _ = self.forward(x)
        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    @staticmethod
    def _normal_log_density(x, mean, log_std, std):
        var = std.pow(2)
        log_density = -(x - mean).pow(2) / (2 * var) - \
            0.5 * math.log(2 * math.pi) - log_std
        return log_density.sum(1, keepdim=True)


class ActorCriticMLP(PGSupportingPolicy):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 is_disc_action: bool,
                 hidden_size=64,
                 embedding: Optional[Callable] = None,
                 activation: Callable = F.elu,
                 log_std: int = 0,
                 **kwargs):

        super().__init__(obs_dim, action_dim, is_disc_action, **kwargs)

        def init(module, weight_init, bias_init, gain=1):
            weight_init(module.weight.data, gain=gain)
            bias_init(module.bias.data)
            return module

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(obs_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(obs_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.action_mean = nn.Linear(hidden_size, action_dim)
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

        self.train()

    def forward(self, x):

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        action_mean = self.action_mean(hidden_actor)

        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std, self.critic_linear(hidden_critic)

    def get_value(self, x):
        _, _, _, val = self.forward(x)
        return val

    def select_action(self, x: Tensor) -> float:
        action_mean, _, action_std, _ = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action.detach().cpu().numpy()[0]

    def get_log_prob(self, x: Tensor, actions: Tensor) -> Tensor:
        action_mean, action_log_std, action_std, _ = self.forward(x)
        return self._normal_log_density(
            actions, action_mean, action_log_std, action_std)

    def get_kl(self, x: torch.Tensor):
        mean1, log_std1, std1, _ = self.forward(x)
        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    @staticmethod
    def _normal_log_density(x, mean, log_std, std):
        var = std.pow(2)
        log_density = -(x - mean).pow(2) / (2 * var) - \
            0.5 * math.log(2 * math.pi) - log_std
        return log_density.sum(1, keepdim=True)
