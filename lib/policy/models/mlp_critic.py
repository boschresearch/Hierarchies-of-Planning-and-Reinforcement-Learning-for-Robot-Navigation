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

import torch.nn as nn
import torch.nn.functional as F


class StateValue(nn.Module):
    def __init__(self, state_dim, hidden_size=(64, 64, 64), activation='relu'):
        super().__init__()
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value
