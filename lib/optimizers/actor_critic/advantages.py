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


def gae(rewards, masks, values, discount, gae_lambda, use_gpu=False):
    if use_gpu:
        rewards, masks, values = rewards.cpu(), masks.cpu(), values.cpu()

    tensor_type = type(rewards)
    returns = tensor_type(rewards.size(0), 1)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + discount * prev_return * masks[i]
        deltas[i] = rewards[i] + discount * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + discount * gae_lambda * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]
    advantages = (advantages - advantages.mean()) / (advantages.std())

    if use_gpu:
        advantages, returns = advantages.cuda(), returns.cuda()
    return advantages, returns
