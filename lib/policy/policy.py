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


from abc import abstractmethod
from typing import Union

import numpy as np

import torch
import torch.nn as nn

from typing import Tuple, List, Optional


class PytorchPolicy(nn.Module):
    # TODO: Store obs_descriptor_dict also in state_dict and compare when loading parameters
    def __init__(self, obs_dim: int,
                 action_dim: int,
                 is_disc_action: bool,
                 *args,
                 obs_descriptor_dict: Optional[dict]=None,
                 subaction_dims: Optional[Union[Tuple[int], List[int]]]=None,
                 **kwargs):
        """
        Creates a new PytorchPolicy instance
        :param obs_dim: int, size of the flat observation space
        :param action_dim: int, size of the flat action space
        :param is_disc_action: bool, specifies whether the policy's output is a discrete action
        :param args: list, additional subclass arguments
        :param obs_descriptor_dict: Optional[dict], dictionary {semantic: obs_index} describing the
                                    semantics of the observation. If None, an descriptor {'obs_<ix>',ix}
                                    will be auto-generated.
        :param subaction_dims: Optional[Union[Tuple[int], List[int]]], factorization of the action into sub-actions.
                               Default resolves to (obs_dim,)
        :param kwargs: dict, additional subclass key-word arguments
        """
        # We need to call the nn.Module constructor first,
        # because of its parameter and buffer registration
        nn.Module.__init__(self)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.is_disc_action = is_disc_action

        self.subaction_dims = tuple(subaction_dims) if subaction_dims is not None else (action_dim,)
        if int(self.action_dim) != int(np.prod(self.subaction_dims)):
            raise ValueError('Input argument action_dim {} does not match'.format(action_dim)
                             + 'product of key-word argument subaction_dims {}.'.format(int(np.prod(self.subaction_dims))))

        if obs_descriptor_dict is None:
            obs_descriptor_dict = {'obs_{}'.format(ix): ix for ix in range(obs_dim)}
        self.obs_descriptor_dict = obs_descriptor_dict


    @abstractmethod
    def select_action(self, obs: Union[np.ndarray, torch.tensor], *args, **kwargs) -> int:
        """
        Applies the policy for a given observation and time step
        :param obs: an observation vector of the environment
        :return: the chosen action of the policy as well as an info dictionary
        """

    def update_state_stats(self, x: torch.Tensor, *args, **kwargs) -> None:
        """
        Update state statistics in derived classes. Does nothing if not
        overridden.
        :param x: Tensor
        """
        pass


class PGSupportingPolicy(PytorchPolicy):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple:
        # returns a tuple of (result, log_info)
        raise NotImplementedError

    @abstractmethod
    def select_action(self, x: Union[np.ndarray, torch.Tensor]) -> float:
        raise NotImplementedError

    def get_kl(self, x: torch.Tensor):
        action_prob1, _ = self.forward(x)
        action_prob0 = torch.tensor(action_prob1.data)
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x: torch.Tensor, actions: List[int]) -> torch.Tensor:
        """
        Returns list of log(probability of selected action) after a forward pass.
        :param actions: list of selected actions
        """
        log_probs = self.get_log_probs(x)
        return log_probs.gather(1, actions.unsqueeze(1).long())

    def get_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        action_prob, _ = self.forward(x)
        action_prob = action_prob.clamp(min=1e-8)  # ensure all small values won't explode to infinity in backward.
        return torch.log(action_prob)

    def get_fim(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        action_prob, _ = self.forward(x)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}
