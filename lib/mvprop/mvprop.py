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
import torch.nn.functional as F


class MVPROPFAT(nn.Module):
    def __init__(self, k):
        super(MVPROPFAT, self).__init__()
        self.h = nn.Conv2d(in_channels=1, out_channels=150, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.p = nn.Conv2d(in_channels=150, out_channels=1, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.k = k
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, image):
        r = image[:, 1, :, :].unsqueeze(1)
        h = F.relu(self.h(image[:, 0, :, :].unsqueeze(1)))
        p = torch.sigmoid(self.p(h))

        v = r
        for _ in range(self.k):
            v = torch.max(v, r + p * (self.max_pool(v) - r))

        v = torch.where(image[:, 0, :, :].unsqueeze(1) > 0.49, v, -1 * torch.ones_like(v))

        return v


class MVPROPFAT3(nn.Module):
    def __init__(self, k):
        super(MVPROPFAT3, self).__init__()
        self.h = nn.Conv2d(in_channels=2, out_channels=150, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.p = nn.Conv2d(in_channels=150, out_channels=1, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.k = k
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, image):
        r = image[:, 2, :, :].unsqueeze(1)
        h = F.relu(self.h(image[:, :2, :, :]))
        p = torch.sigmoid(self.p(h))

        v = r
        for _ in range(self.k):
            v = torch.max(v, r + p * (self.max_pool(v) - r))

        v = torch.where(image[:, 0, :, :].unsqueeze(1) > 0.49, v, -1 * torch.ones_like(v))

        return v


class MVPROPFAT3D(nn.Module):
    def __init__(self, k):
        super(MVPROPFAT3D, self).__init__()
        self.h = nn.Conv3d(in_channels=1, out_channels=150, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=True)
        self.p = nn.Conv3d(in_channels=150, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)
        self.k = k
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

    def forward(self, image):
        r = image[:, 1, :, :, :].unsqueeze(1)
        h = F.relu(self.h(image[:, 0, :, :, :].unsqueeze(1)))
        p = torch.sigmoid(self.p(h))

        v = r
        for _ in range(self.k):
            v = torch.max(v, r + p * (self.max_pool(v) - r))

        return v
