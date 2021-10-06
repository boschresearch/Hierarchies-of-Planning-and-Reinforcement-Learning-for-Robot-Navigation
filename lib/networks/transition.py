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


import torch.nn as nn
import torch.nn.functional as F


class TransNetFlat(nn.Module):
    def __init__(self, num_out=9):
        super(TransNetFlat, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class TransNetFlat19(nn.Module):
    def __init__(self, num_out=9):
        super(TransNetFlat19, self).__init__()
        self.fc1 = nn.Linear(19, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class TransNetFlatParkingO(nn.Module):
    def __init__(self, num_out_p=9, num_out_o=8):
        super(TransNetFlatParkingO, self).__init__()
        self.fc = nn.Linear(3, 32)
        self.fc_hidden = nn.Linear(32, 32)
        self.fc_pos = nn.Linear(32, num_out_p)
        self.fc_orient = nn.Linear(32, num_out_o)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc_hidden(x))
        pos = self.fc_pos(x)
        orient = self.fc_orient(x)
        return pos, orient

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
