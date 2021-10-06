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


import torch
import torch.nn as nn
import torch.optim as optim


class MVPROPOptimizerTD0:
    def __init__(self, mvprop, target_mvprop, memory, gamma, bs=128, lr=1e-3, k=35):
        self.mvprop = mvprop
        self.optimizer = optim.Adam(params=self.mvprop.parameters(), lr=lr)
        self.target_mvprop = target_mvprop
        self.memory = memory
        self.gamma = gamma
        self.bs = bs
        self.lr = lr
        self.k = k
        self.loss_func = nn.MSELoss()

    def train(self, horizon):
        z, o, zp, r, nd, im = self.memory.sample(self.bs)

        with torch.no_grad():
            v = self.target_mvprop(im)
            v = v.detach()
            v_select_target = v[list(range(self.bs)), 0, list(zp.numpy()[:, 0]), list(zp.numpy()[:, 1])]

        v = self.mvprop(im)
        v_select = v[list(range(self.bs)), 0, list(z.numpy()[:, 0]), list(z.numpy()[:, 1])]

        pred_val = v_select * horizon
        target_val = r[:, 0] + nd[:, 0] * self.gamma * v_select_target * horizon
        loss = self.loss_func(pred_val, target_val)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class MVPROPOptimizer3D:
    def __init__(self, mvprop, target_mvprop, memory, gamma, bs=128, lr=1e-3, k=35):
        self.mvprop = mvprop
        self.optimizer = optim.Adam(params=self.mvprop.parameters(), lr=lr)
        self.target_mvprop = target_mvprop
        self.memory = memory
        self.gamma = gamma
        self.bs = bs
        self.lr = lr
        self.k = k
        self.loss_func = nn.MSELoss()

    def train(self, horizon):
        z, o, zp, r, nd, im = self.memory.sample(self.bs)

        with torch.no_grad():
            v = self.target_mvprop(im)
            v = v.detach()
            v_select_target = v[list(range(self.bs)), 0, list(zp.numpy()[:, 0]), list(zp.numpy()[:, 1]), list(zp.numpy()[:, 2])]

        v = self.mvprop(im)
        v_select = v[list(range(self.bs)), 0, list(z.numpy()[:, 0]), list(z.numpy()[:, 1]), list(z.numpy()[:, 2])]

        pred_val = v_select * horizon
        target_val = r[:, 0] + nd[:, 0] * self.gamma * v_select_target * horizon
        loss = self.loss_func(pred_val, target_val)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
