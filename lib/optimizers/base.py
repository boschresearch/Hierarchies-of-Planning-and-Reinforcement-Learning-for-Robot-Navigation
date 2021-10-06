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


import numpy as np
import torch


class Optimizer(object):
    def __init__(self, policy, use_gpu=False):
        self.networks = self._init_networks(policy.obs_dim, policy.action_dim)

        networks = self.networks.copy()
        networks['policy'] = policy
        self.optimizers = self._init_optimizers(networks)

        self.use_gpu = use_gpu
        if self.use_gpu:
            self.networks = {k: v.cuda() for k, v in self.networks.items()}

    @classmethod
    def _init_networks(cls, obs_dim, action_dim):
        raise NotImplementedError

    def process_batch(self, policy, batch, update_policy_args):
        states, actions, rewards, masks = self._unpack_batch(batch)
        if self.use_gpu:
            states, actions, rewards, masks = map(
                lambda x: x.cuda(), [states, actions, rewards, masks])

        policy = self._update_networks(
            policy, actions, masks, rewards, states,
            batch["num_episodes"], *update_policy_args)
        return policy

    @staticmethod
    def _unpack_batch(batch):
        states = torch.from_numpy(np.array(batch["states"], dtype=np.float32))
        rewards = torch.from_numpy(np.array(batch["rewards"], dtype=np.float32))
        masks = torch.from_numpy(np.array(batch["masks"], dtype=np.float32))
        actions = torch.from_numpy(np.array(batch["actions"]))
        return states, actions, rewards, masks # , weights

    def _update_networks(self, policy,
                         actions, masks, rewards, states, num_episodes,
                         *args, **step_kwargs):
        raise NotImplementedError

    def _init_optimizers(self, networks, lr_rates=None):
        args = {key: [network] for key, network in networks.items()}
        if lr_rates is not None:
            for key in args.keys():
                args[key].append(lr_rates[key])

        optimizers = {key: self._init_optimizer(*args[key])
                      for key in networks.keys()}
        return optimizers

    @staticmethod
    def _init_optimizer(network, lr_rate=0.01):
        return torch.optim.Adam(network.parameters(), lr=lr_rate)


def unpack_batch_standalone(batch):
    states = torch.from_numpy(np.array(batch["states"], dtype=np.float32))
    rewards = torch.from_numpy(np.array(batch["rewards"], dtype=np.float32))
    masks = torch.from_numpy(np.array(batch["masks"], dtype=np.float32))
    actions = torch.from_numpy(np.array(batch["actions"]))
    return states, actions, rewards, masks
