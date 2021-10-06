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

import os

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from typing import Generator

DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor


def get_flat_params_from(model: Module) -> Tensor:
    params = []
    for param in model.parameters():
        params.append(param.detach().contiguous().view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model: Module, flat_params: DoubleTensor) -> None:
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.detach().copy_(
            flat_params[prev_ind:prev_ind + flat_size].contiguous().view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(inputs: Generator, grad_grad: bool = False) -> Tensor:
    grads = []
    for param in inputs:
        if grad_grad:
            grads.append(param.grad.grad.contiguous().view(-1))
        else:
            if param.grad is None:
                grads.append(torch.zeros(param.detach().contiguous().view(-1).shape))
            else:
                grads.append(param.grad.contiguous().view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


def compute_flat_grad(output: Tensor, inputs: Generator,
                      filter_input_ids: set = set(),
                      retain_graph: bool = False,
                      create_graph: bool = False) -> Tensor:
    if create_graph:
        retain_graph = True

    inputs = list(inputs)
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = torch.autograd.grad(output, params, retain_graph=retain_graph,
                                create_graph=create_graph)

    j = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads.append(torch.zeros(param.data.contiguous().view(-1).shape))
        else:
            out_grads.append(grads[j].contiguous().view(-1))
            j += 1
    grads = torch.cat(out_grads)

    for param in params:
        param.grad = None
    return grads
