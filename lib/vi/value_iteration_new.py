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


import copy
import numpy as np
from random import randint
import random

import torch
import torch.nn as nn
import torch.optim as optim

from lib.networks.transition import TransNetFlat, TransNetFlat19


class ValueIteration4M1SyncNN:

    def __init__(self, grid_size_x, grid_size_y, gamma=0.99):
        self.grid_dimensions = (grid_size_x, grid_size_y)
        self.gamma = gamma

        self.P_net = TransNetFlat(num_out=5)
        self.P_table = np.zeros((grid_size_x, grid_size_y, 4, 5))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.P_net.parameters(), lr=0.001)
        self.target = None
        self.stay_prob = 0.5

    def generate_dataset_flat(self, occupancy_map, s_bar_collection, a_bar_collection):

        total_len = len(a_bar_collection)

        occupancy_map_padded = np.ones((self.grid_dimensions[0]+2, self.grid_dimensions[1]+2))
        occupancy_map_padded[1:self.grid_dimensions[0]+1, 1:self.grid_dimensions[1]+1] = occupancy_map

        x_arr = np.zeros((total_len, 10))
        y_arr = np.zeros((total_len, 1))
        w_arr = np.zeros((total_len, 5))

        for idx in range(total_len):
            s_bar = s_bar_collection[idx]
            x_arr[idx, :9] = occupancy_map_padded[s_bar[0]:s_bar[0] + 3, s_bar[1]:s_bar[1] + 3].flatten()
            x_arr[idx, 9] = a_bar_collection[idx]

            s_new_bar = s_bar_collection[idx+1]
            state_diff_x = round(s_new_bar[0] - s_bar[0])
            state_diff_y = round(s_new_bar[1] - s_bar[1])
            if state_diff_x == 0 and state_diff_y == 1:
                diff_idx = 1
                w_arr[idx, 1] = 1
            elif state_diff_x == 1 and state_diff_y == 0:
                diff_idx = 2
                w_arr[idx, 2] = 1
            elif state_diff_x == 0 and state_diff_y == -1:
                diff_idx = 3
                w_arr[idx, 3] = 1
            elif state_diff_x == -1 and state_diff_y == 0:
                diff_idx = 4
                w_arr[idx, 4] = 1
            else:
                diff_idx = 0
                w_arr[idx, 0] = 1

            y_arr[idx] = diff_idx

        return x_arr, y_arr, w_arr

    def generate_dataset_flat_agents(self, occupancy_map_list, s_bar_collection, a_bar_collection):

        total_len = len(a_bar_collection)

        x_arr = np.zeros((total_len, 10))
        y_arr = np.zeros((total_len, 1))
        w_arr = np.zeros((total_len, 5))

        for idx in range(total_len):
            s_bar = s_bar_collection[idx]
            occupancy_map_padded = np.ones((self.grid_dimensions[0] + 2, self.grid_dimensions[1] + 2))
            occupancy_map_padded[1:self.grid_dimensions[0] + 1, 1:self.grid_dimensions[1] + 1] = occupancy_map_list[idx]
            x_arr[idx, :9] = occupancy_map_padded[s_bar[0]:s_bar[0] + 3, s_bar[1]:s_bar[1] + 3].flatten()
            x_arr[idx, 9] = a_bar_collection[idx]

            s_new_bar = s_bar_collection[idx+1]
            state_diff_x = round(s_new_bar[0] - s_bar[0])
            state_diff_y = round(s_new_bar[1] - s_bar[1])
            if state_diff_x == 0 and state_diff_y == 1:
                diff_idx = 1
                w_arr[idx, 1] = 1
            elif state_diff_x == 1 and state_diff_y == 0:
                diff_idx = 2
                w_arr[idx, 2] = 1
            elif state_diff_x == 0 and state_diff_y == -1:
                diff_idx = 3
                w_arr[idx, 3] = 1
            elif state_diff_x == -1 and state_diff_y == 0:
                diff_idx = 4
                w_arr[idx, 4] = 1
            else:
                diff_idx = 0
                w_arr[idx, 0] = 1

            y_arr[idx] = diff_idx

        return x_arr, y_arr, w_arr

    def train_net(self, buffer, bs, opt_iterations, rw=False):

        if rw:
            self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(buffer.get_weights()))
        else:
            self.criterion = nn.CrossEntropyLoss()

        for ijk in range(opt_iterations):
            x, y = buffer.sample(batch_size=bs)
            x_batch = torch.from_numpy(np.float32(x))
            y_batch = torch.from_numpy(np.int64(y))

            self.optimizer.zero_grad()

            outputs = self.P_net(x_batch)
            loss = self.criterion(outputs, y_batch.squeeze())
            loss.backward()
            self.optimizer.step()

    def update_p_table_optimistic(self, occupancy_map, walls=False):
        self.P_table[:, :, :, :] = 0.
        if walls:
            for idx_x in range(self.grid_dimensions[0]):
                for idx_y in range(self.grid_dimensions[1]):
                    for idx_a in range(4):
                        if idx_a == 0:
                            if (idx_y + 1 < self.grid_dimensions[1]) and (occupancy_map[idx_x, idx_y + 1] == 0):
                                self.P_table[idx_x, idx_y, 0, 1] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 0, 0] = 1.0
                        elif idx_a == 1:
                            if (idx_x + 1 < self.grid_dimensions[0]) and (occupancy_map[idx_x + 1, idx_y] == 0):
                                self.P_table[idx_x, idx_y, 1, 2] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 1, 0] = 1.0
                        elif idx_a == 2:
                            if (idx_y - 1 > 0) and (occupancy_map[idx_x, idx_y - 1] == 0):
                                self.P_table[idx_x, idx_y, 2, 3] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 2, 0] = 1.0
                        elif idx_a == 3:
                            if (idx_x - 1 > 0) and (occupancy_map[idx_x - 1, idx_y] == 0):
                                self.P_table[idx_x, idx_y, 3, 4] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 3, 0] = 1.0
        else:
            for idx_x in range(self.grid_dimensions[0]):
                for idx_y in range(self.grid_dimensions[1]):
                    for idx_a in range(4):
                        self.P_table[idx_x, idx_y, idx_a, idx_a + 1] = 1.0

    def update_stay_prob(self, stay_prob_new, alpha=0.1):
        self.stay_prob = alpha * stay_prob_new + (1 - alpha) * self.stay_prob

    def update_p_table(self, occupancy_map, walls=False, stay=False):
        occupancy_map_padded = np.ones((self.grid_dimensions[0]+2, self.grid_dimensions[1]+2))
        occupancy_map_padded[1:self.grid_dimensions[0]+1, 1:self.grid_dimensions[1]+1] = occupancy_map
        for idx_x in range(self.grid_dimensions[0]):
            for idx_y in range(self.grid_dimensions[1]):
                for idx_a in range(4):
                    x = np.zeros((10, ))
                    x[:9] = occupancy_map_padded[idx_x:idx_x + 3, idx_y:idx_y + 3].flatten()
                    x[9] = idx_a
                    softmax = torch.nn.Softmax(dim=1)
                    if stay:
                        self.P_table[idx_x, idx_y, idx_a, :] = np.array([self.stay_prob, 0, 0, 0, 0, 0, 0, 0, 0]) + \
                                                               (1 - self.stay_prob) * softmax(
                            self.P_net(torch.from_numpy(np.float32(np.expand_dims(x, axis=0))))).data.numpy()
                    else:
                        self.P_table[idx_x, idx_y, idx_a, :] = softmax(self.P_net(torch.from_numpy(np.float32(np.expand_dims(x, axis=0))))).data.numpy()

                if walls and (occupancy_map[idx_x, idx_y] == 1):
                    self.P_table[idx_x, idx_y, :, :] = 0

    def run_vi(self, grid, goal, num_runs=100, V_init=None):

        if V_init is not None:
            V = copy.deepcopy(V_init)
            V_old = copy.deepcopy(V_init)
        else:
            V = np.zeros(self.grid_dimensions)
            V_old = np.zeros(self.grid_dimensions)
        PI = np.zeros(self.grid_dimensions)

        R = -1 * np.ones(self.grid_dimensions)
        R[goal[0], goal[1]] = 0.
        R_tensor = np.zeros((grid.x_size + 2, grid.y_size + 2, 5))
        rows, columns, _ = R_tensor.shape
        R_tensor[1:rows - 1, 1:columns - 1, 0] = R
        R_tensor[1:rows - 1, 0:columns - 2, 1] = R
        R_tensor[0:rows - 2, 1:columns - 1, 2] = R
        R_tensor[1:rows - 1, 2:columns, 3] = R
        R_tensor[2:rows, 1:columns - 1, 4] = R
        R_tensor = R_tensor[1:rows - 1, 1:columns - 1, :]
        R_tensor = np.repeat(np.expand_dims(R_tensor, axis=2), 4, axis=2)

        P = copy.deepcopy(self.P_table)
        P[goal[0], goal[1], :, :] = np.zeros((4, 5))
        P[goal[0], goal[1], :, 0] = 1.0  # new

        mask = np.zeros((grid.x_size + 2, grid.y_size + 2, 5))
        mask[1:rows - 1, 1:columns - 1, 0] = grid.occupancy_map_un_padded
        mask[1:rows - 1, 0:columns - 2, 1] = grid.occupancy_map_un_padded
        mask[0:rows - 2, 1:columns - 1, 2] = grid.occupancy_map_un_padded
        mask[1:rows - 1, 2:columns, 3] = grid.occupancy_map_un_padded
        mask[2:rows, 1:columns - 1, 4] = grid.occupancy_map_un_padded
        mask = mask[1:rows - 1, 1:columns - 1, :]
        mask = np.repeat(np.expand_dims(mask, axis=2), 4, axis=2)

        vi_iter = 0

        for _ in range(num_runs):
            vi_iter += 1
            V_rep = np.zeros((grid.x_size + 2, grid.y_size + 2, 5))
            V_rep[1:rows - 1, 1:columns - 1, 0] = V
            V_rep[1:rows - 1, 0:columns - 2, 1] = V
            V_rep[0:rows - 2, 1:columns - 1, 2] = V
            V_rep[1:rows - 1, 2:columns, 3] = V
            V_rep[2:rows, 1:columns - 1, 4] = V
            V_rep = V_rep[1:rows - 1, 1:columns - 1, :]
            V_rep = np.repeat(np.expand_dims(V_rep, axis=2), 4, axis=2)

            V_prop = P * (R_tensor*mask + self.gamma*V_rep*mask)

            V_a = np.sum(V_prop, axis=3)

            V = np.max(V_a, axis=2)

            progress = (np.abs(V - V_old)).mean()
            V_old = copy.deepcopy(V)
            if progress < 1e-3:
                break

        for ix in range(grid.x_size):
            for iy in range(grid.y_size):
                if (np.max(V_a[ix, iy, :]) == 0) and (abs(V_a[ix, iy, :].mean() - V_a[ix, iy, 0]) < 1e-5):
                    PI[ix, iy] = randint(0, 3)
                else:
                    PI[ix, iy] = np.argmax(V_a[ix, iy, :])
        return V, PI

    def set_target(self, tile, action, mode='normal', occupancy_map=None):
        if mode == 'random':
            applicable = False
            while not applicable:
                action = random.randint(0, 3)
                if action == 0:
                    if (tile[1] + 1 < self.grid_dimensions[1]) and (occupancy_map[tile[0], tile[1] + 1] == 0):
                        applicable = True
                        self.target = (tile[0], tile[1] + 1)
                elif action == 1:
                    if (tile[0] + 1 < self.grid_dimensions[0]) and (occupancy_map[tile[0] + 1, tile[1]] == 0):
                        applicable = True
                        self.target = (tile[0] + 1, tile[1])
                elif action == 2:
                    if (tile[1] - 1 > 0) and (occupancy_map[tile[0], tile[1] - 1] == 0):
                        applicable = True
                        self.target = (tile[0], tile[1] - 1)
                elif action == 3:
                    if (tile[0] - 1 > 0) and (occupancy_map[tile[0] - 1, tile[1]] == 0):
                        applicable = True
                        self.target = (tile[0] - 1, tile[1])
        else:
            if action == 0:
                self.target = (tile[0], tile[1] + 1)
            elif action == 1:
                self.target = (tile[0] + 1, tile[1])
            elif action == 2:
                self.target = (tile[0], tile[1] - 1)
            elif action == 3:
                self.target = (tile[0] - 1, tile[1])
        self.target = (max(0, min(self.target[0], self.grid_dimensions[0] - 1)),
                       max(0, min(self.target[1], self.grid_dimensions[1] - 1)))

    def get_target_vec(self, state):
        return self.target[0] - state[0], self.target[1] - state[1]

    def get_target_vec_t2(self, state):
        return 0.5*self.target[0] - state[0], 0.5*self.target[1] - state[1]

    def get_target_vec_mj_point(self, state):
        return 0.25*self.target[0] - 0.875 - state[0], 0.25*self.target[1] - 0.875 - state[1]

    def get_target_vec_mj_ant(self, state):
        return self.target[0] - 1.0 - state[0], self.target[1] - 1.0 - state[1]

    def get_target(self):
        return self.target


class ValueIteration4M1SyncNN8:

    def __init__(self, grid_size_x, grid_size_y, gamma=0.99):
        self.grid_dimensions = (grid_size_x, grid_size_y)
        self.gamma = gamma

        self.P_net = TransNetFlat()
        self.P_table = np.zeros((grid_size_x, grid_size_y, 8, 9))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.P_net.parameters(), lr=0.001)
        self.target = None
        self.stay_prob = 0.5

    def generate_dataset_flat(self, occupancy_map, s_bar_collection, a_bar_collection):

        total_len = len(a_bar_collection)

        occupancy_map_padded = np.ones((self.grid_dimensions[0]+2, self.grid_dimensions[1]+2))
        occupancy_map_padded[1:self.grid_dimensions[0]+1, 1:self.grid_dimensions[1]+1] = occupancy_map

        x_arr = np.zeros((total_len, 10))
        y_arr = np.zeros((total_len, 1))
        w_arr = np.zeros((total_len, 9))

        for idx in range(total_len):
            s_bar = s_bar_collection[idx]
            x_arr[idx, :9] = occupancy_map_padded[s_bar[0]:s_bar[0] + 3, s_bar[1]:s_bar[1] + 3].flatten()
            x_arr[idx, 9] = a_bar_collection[idx]

            s_new_bar = s_bar_collection[idx+1]
            state_diff_x = np.clip(round(s_new_bar[0] - s_bar[0]), -1, 1, dtype=np.int)
            state_diff_y = np.clip(round(s_new_bar[1] - s_bar[1]), -1, 1, dtype=np.int)
            if state_diff_x == 0 and state_diff_y == 1:
                diff_idx = 1
                w_arr[idx, 1] = 1
            elif state_diff_x == 1 and state_diff_y == 1:
                diff_idx = 2
                w_arr[idx, 2] = 1
            elif state_diff_x == 1 and state_diff_y == 0:
                diff_idx = 3
                w_arr[idx, 3] = 1
            elif state_diff_x == 1 and state_diff_y == -1:
                diff_idx = 4
                w_arr[idx, 4] = 1
            elif state_diff_x == 0 and state_diff_y == -1:
                diff_idx = 5
                w_arr[idx, 5] = 1
            elif state_diff_x == -1 and state_diff_y == -1:
                diff_idx = 6
                w_arr[idx, 6] = 1
            elif state_diff_x == -1 and state_diff_y == 0:
                diff_idx = 7
                w_arr[idx, 7] = 1
            elif state_diff_x == -1 and state_diff_y == 1:
                diff_idx = 8
                w_arr[idx, 8] = 1
            else:
                diff_idx = 0
                w_arr[idx, 0] = 1

            y_arr[idx] = diff_idx

        return x_arr, y_arr, w_arr

    def generate_dataset_flat_agents(self, occupancy_map_list, s_bar_collection, a_bar_collection):

        total_len = len(a_bar_collection)

        x_arr = np.zeros((total_len, 10))
        y_arr = np.zeros((total_len, 1))
        w_arr = np.zeros((total_len, 9))

        for idx in range(total_len):
            s_bar = s_bar_collection[idx]
            occupancy_map_padded = np.ones((self.grid_dimensions[0] + 2, self.grid_dimensions[1] + 2))
            occupancy_map_padded[1:self.grid_dimensions[0] + 1, 1:self.grid_dimensions[1] + 1] = occupancy_map_list[idx]
            x_arr[idx, :9] = occupancy_map_padded[s_bar[0]:s_bar[0] + 3, s_bar[1]:s_bar[1] + 3].flatten()
            x_arr[idx, 9] = a_bar_collection[idx]

            s_new_bar = s_bar_collection[idx + 1]
            state_diff_x = np.clip(round(s_new_bar[0] - s_bar[0]), -1, 1, dtype=np.int)
            state_diff_y = np.clip(round(s_new_bar[1] - s_bar[1]), -1, 1, dtype=np.int)
            if state_diff_x == 0 and state_diff_y == 1:
                diff_idx = 1
                w_arr[idx, 1] = 1
            elif state_diff_x == 1 and state_diff_y == 1:
                diff_idx = 2
                w_arr[idx, 2] = 1
            elif state_diff_x == 1 and state_diff_y == 0:
                diff_idx = 3
                w_arr[idx, 3] = 1
            elif state_diff_x == 1 and state_diff_y == -1:
                diff_idx = 4
                w_arr[idx, 4] = 1
            elif state_diff_x == 0 and state_diff_y == -1:
                diff_idx = 5
                w_arr[idx, 5] = 1
            elif state_diff_x == -1 and state_diff_y == -1:
                diff_idx = 6
                w_arr[idx, 6] = 1
            elif state_diff_x == -1 and state_diff_y == 0:
                diff_idx = 7
                w_arr[idx, 7] = 1
            elif state_diff_x == -1 and state_diff_y == 1:
                diff_idx = 8
                w_arr[idx, 8] = 1
            else:
                diff_idx = 0
                w_arr[idx, 0] = 1

            y_arr[idx] = diff_idx

        return x_arr, y_arr, w_arr

    def train_net(self, buffer, bs, opt_iterations, rw=False):

        if rw:
            self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(buffer.get_weights()))
        else:
            self.criterion = nn.CrossEntropyLoss()

        for ijk in range(opt_iterations):
            x, y = buffer.sample(batch_size=bs)
            x_batch = torch.from_numpy(np.float32(x))
            y_batch = torch.from_numpy(np.int64(y))

            self.optimizer.zero_grad()

            outputs = self.P_net(x_batch)
            loss = self.criterion(outputs, y_batch.squeeze())
            loss.backward()
            self.optimizer.step()

    def update_p_table_optimistic(self, occupancy_map, walls=False):
        self.P_table[:, :, :, :] = 0.
        if walls:
            for idx_x in range(self.grid_dimensions[0]):
                for idx_y in range(self.grid_dimensions[1]):
                    for idx_a in range(8):
                        if idx_a == 0:
                            if (idx_y + 1 < self.grid_dimensions[1]) and (occupancy_map[idx_x, idx_y + 1] == 0):
                                self.P_table[idx_x, idx_y, 0, 1] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 0, 0] = 1.0
                        elif idx_a == 1:
                            if (idx_x + 1 < self.grid_dimensions[0]) and (idx_y + 1 < self.grid_dimensions[1]) and \
                                    (occupancy_map[idx_x + 1, idx_y + 1] == 0):
                                self.P_table[idx_x, idx_y, 1, 2] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 1, 0] = 1.0
                        elif idx_a == 2:
                            if (idx_x + 1 < self.grid_dimensions[0]) and (occupancy_map[idx_x + 1, idx_y] == 0):
                                self.P_table[idx_x, idx_y, 2, 3] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 2, 0] = 1.0
                        elif idx_a == 3:
                            if (idx_x + 1 < self.grid_dimensions[0]) and (idx_y - 1 > 0) and \
                                    (occupancy_map[idx_x + 1, idx_y - 1] == 0):
                                self.P_table[idx_x, idx_y, 3, 4] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 3, 0] = 1.0
                        elif idx_a == 4:
                            if (idx_y - 1 > 0) and (occupancy_map[idx_x, idx_y - 1] == 0):
                                self.P_table[idx_x, idx_y, 4, 5] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 4, 0] = 1.0
                        elif idx_a == 5:
                            if (idx_x - 1 > 0) and (idx_y - 1 > 0) and \
                                    (occupancy_map[idx_x - 1, idx_y - 1] == 0):
                                self.P_table[idx_x, idx_y, 5, 6] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 5, 0] = 1.0
                        elif idx_a == 6:
                            if (idx_x - 1 > 0) and (occupancy_map[idx_x - 1, idx_y] == 0):
                                self.P_table[idx_x, idx_y, 6, 7] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 6, 0] = 1.0
                        elif idx_a == 7:
                            if (idx_x - 1 > 0) and (idx_y + 1 < self.grid_dimensions[1]) and \
                                    (occupancy_map[idx_x - 1, idx_y + 1] == 0):
                                self.P_table[idx_x, idx_y, 7, 8] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 7, 0] = 1.0
        else:
            for idx_x in range(self.grid_dimensions[0]):
                for idx_y in range(self.grid_dimensions[1]):
                    for idx_a in range(8):
                        self.P_table[idx_x, idx_y, idx_a, idx_a + 1] = 1.0

    def update_stay_prob(self, stay_prob_new, alpha=0.1):
        self.stay_prob = alpha * stay_prob_new + (1 - alpha) * self.stay_prob

    def update_p_table(self, occupancy_map, walls=False):
        occupancy_map_padded = np.ones((self.grid_dimensions[0] + 2, self.grid_dimensions[1] + 2))
        occupancy_map_padded[1:self.grid_dimensions[0] + 1, 1:self.grid_dimensions[1] + 1] = occupancy_map
        for idx_x in range(self.grid_dimensions[0]):
            for idx_y in range(self.grid_dimensions[1]):
                for idx_a in range(8):
                    x = np.zeros((10,))
                    x[:9] = occupancy_map_padded[idx_x:idx_x + 3, idx_y:idx_y + 3].flatten()
                    x[9] = idx_a
                    softmax = torch.nn.Softmax(dim=1)
                    self.P_table[idx_x, idx_y, idx_a, :] = softmax(
                        self.P_net(torch.from_numpy(np.float32(np.expand_dims(x, axis=0))))).data.numpy()

                if walls and (occupancy_map[idx_x, idx_y] == 1):
                    self.P_table[idx_x, idx_y, :, :] = 0

    def run_vi(self, grid, goal, num_runs=100, V_init=None):

        if V_init is not None:
            V = copy.deepcopy(V_init)
            V_old = copy.deepcopy(V_init)
        else:
            V = np.zeros(self.grid_dimensions)
            V_old = np.zeros(self.grid_dimensions)
        PI = np.zeros(self.grid_dimensions)

        R = -1 * np.ones(self.grid_dimensions)
        R[goal[0], goal[1]] = 0.
        R_tensor = np.zeros((grid.x_size + 2, grid.y_size + 2, 9))
        rows, columns, _ = R_tensor.shape
        R_tensor[1:rows - 1, 1:columns - 1, 0] = R
        R_tensor[1:rows - 1, 0:columns - 2, 1] = R
        R_tensor[0:rows - 2, 0:columns - 2, 2] = R
        R_tensor[0:rows - 2, 1:columns - 1, 3] = R
        R_tensor[0:rows - 2, 2:columns, 4] = R
        R_tensor[1:rows - 1, 2:columns, 5] = R
        R_tensor[2:rows, 2:columns, 6] = R
        R_tensor[2:rows, 1:columns - 1, 7] = R
        R_tensor[2:rows, 0:columns - 2, 8] = R
        R_tensor = R_tensor[1:rows - 1, 1:columns - 1, :]
        R_tensor = np.repeat(np.expand_dims(R_tensor, axis=2), 8, axis=2)

        P = copy.deepcopy(self.P_table)
        P[goal[0], goal[1], :, :] = np.zeros((8, 9))
        P[goal[0], goal[1], :, 0] = 1.0  # new

        mask = np.zeros((grid.x_size + 2, grid.y_size + 2, 9))
        mask[1:rows - 1, 1:columns - 1, 0] = grid.occupancy_map_un_padded
        mask[1:rows - 1, 0:columns - 2, 1] = grid.occupancy_map_un_padded
        mask[0:rows - 2, 0:columns - 2, 2] = grid.occupancy_map_un_padded
        mask[0:rows - 2, 1:columns - 1, 3] = grid.occupancy_map_un_padded
        mask[0:rows - 2, 2:columns, 4] = grid.occupancy_map_un_padded
        mask[1:rows - 1, 2:columns, 5] = grid.occupancy_map_un_padded
        mask[2:rows, 2:columns, 6] = grid.occupancy_map_un_padded
        mask[2:rows, 1:columns - 1, 7] = grid.occupancy_map_un_padded
        mask[2:rows, 0:columns - 2, 8] = grid.occupancy_map_un_padded
        mask = mask[1:rows - 1, 1:columns - 1, :]
        mask = np.repeat(np.expand_dims(mask, axis=2), 8, axis=2)

        vi_iter = 0

        for _ in range(num_runs):
            vi_iter += 1
            V_rep = np.zeros((grid.x_size + 2, grid.y_size + 2, 9))
            V_rep[1:rows - 1, 1:columns - 1, 0] = V
            V_rep[1:rows - 1, 0:columns - 2, 1] = V
            V_rep[0:rows - 2, 0:columns - 2, 2] = V
            V_rep[0:rows - 2, 1:columns - 1, 3] = V
            V_rep[0:rows - 2, 2:columns, 4] = V
            V_rep[1:rows - 1, 2:columns, 5] = V
            V_rep[2:rows, 2:columns, 6] = V
            V_rep[2:rows, 1:columns - 1, 7] = V
            V_rep[2:rows, 0:columns - 2, 8] = V
            V_rep = V_rep[1:rows - 1, 1:columns - 1, :]
            V_rep = np.repeat(np.expand_dims(V_rep, axis=2), 8, axis=2)

            V_prop = P * (R_tensor*mask + self.gamma*V_rep*mask)

            V_a = np.sum(V_prop, axis=3)

            V = np.max(V_a, axis=2)

            progress = (np.abs(V - V_old)).mean()
            V_old = copy.deepcopy(V)
            if progress < 1e-3:
                break

        for ix in range(grid.x_size):
            for iy in range(grid.y_size):
                if (np.max(V_a[ix, iy, :]) == 0) and (abs(V_a[ix, iy, :].mean() - V_a[ix, iy, 0]) < 1e-5):
                    PI[ix, iy] = randint(0, 7)
                else:
                    PI[ix, iy] = np.argmax(V_a[ix, iy, :])

        return V, PI

    def set_target(self, tile, action, mode='normal', occupancy_map=None):
        if mode == 'random':
            applicable = False
            while not applicable:
                action = random.randint(0, 7)
                if action == 0:
                    if (tile[1] + 1 < self.grid_dimensions[1]) and (occupancy_map[tile[0], tile[1] + 1] == 0):
                        applicable = True
                        self.target = (tile[0], tile[1] + 1)
                elif action == 1:
                    if (tile[1] + 1 < self.grid_dimensions[1]) and (tile[0] + 1 < self.grid_dimensions[0]) and \
                            (occupancy_map[tile[0] + 1, tile[1] + 1] == 0):
                        applicable = True
                        self.target = (tile[0] + 1, tile[1] + 1)
                elif action == 2:
                    if (tile[0] + 1 < self.grid_dimensions[0]) and (occupancy_map[tile[0] + 1, tile[1]] == 0):
                        applicable = True
                        self.target = (tile[0] + 1, tile[1])
                elif action == 3:
                    if (tile[0] + 1 < self.grid_dimensions[0]) and (tile[1] - 1 > 0) and \
                            (occupancy_map[tile[0] + 1, tile[1] - 1] == 0):
                        applicable = True
                        self.target = (tile[0] + 1, tile[1] - 1)
                elif action == 4:
                    if (tile[1] - 1 > 0) and (occupancy_map[tile[0], tile[1] - 1] == 0):
                        applicable = True
                        self.target = (tile[0], tile[1] - 1)
                elif action == 5:
                    if (tile[0] - 1 > 0) and (tile[1] - 1 > 0) and (occupancy_map[tile[0] - 1, tile[1] - 1] == 0):
                        applicable = True
                        self.target = (tile[0] - 1, tile[1] - 1)
                elif action == 6:
                    if (tile[0] - 1 > 0) and (occupancy_map[tile[0] - 1, tile[1]] == 0):
                        applicable = True
                        self.target = (tile[0] - 1, tile[1])
                elif action == 7:
                    if (tile[1] + 1 < self.grid_dimensions[1]) and (tile[0] - 1 > 0) and \
                            (occupancy_map[tile[0] - 1, tile[1] + 1] == 0):
                        applicable = True
                        self.target = (tile[0] - 1, tile[1] + 1)
        else:
            if action == 0:
                self.target = (tile[0], tile[1] + 1)
            elif action == 1:
                self.target = (tile[0] + 1, tile[1] + 1)
            elif action == 2:
                self.target = (tile[0] + 1, tile[1])
            elif action == 3:
                self.target = (tile[0] + 1, tile[1] - 1)
            elif action == 4:
                self.target = (tile[0], tile[1] - 1)
            elif action == 5:
                self.target = (tile[0] - 1, tile[1] - 1)
            elif action == 6:
                self.target = (tile[0] - 1, tile[1])
            elif action == 7:
                self.target = (tile[0] - 1, tile[1] + 1)
        self.target = (max(0, min(self.target[0], self.grid_dimensions[0] - 1)),
                       max(0, min(self.target[1], self.grid_dimensions[1] - 1)))

    def get_target_vec(self, state):
        return self.target[0] - state[0], self.target[1] - state[1]

    def get_target_vec_t2(self, state):
        return 0.5*self.target[0] - state[0], 0.5*self.target[1] - state[1]

    def get_target_vec_mj_point(self, state):
        return 0.25*self.target[0] - 0.875 - state[0], 0.25*self.target[1] - 0.875 - state[1]

    def get_target_vec_mj_ant(self, state):
        return self.target[0] - 1.0 - state[0], self.target[1] - 1.0 - state[1]

    def get_target(self):
        return self.target


class ValueIteration4M1SyncNN819:

    def __init__(self, grid_size_x, grid_size_y, gamma=0.99):
        self.grid_dimensions = (grid_size_x, grid_size_y)
        self.gamma = gamma

        self.P_net = TransNetFlat19()
        self.P_table = np.zeros((grid_size_x, grid_size_y, 8, 9))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.P_net.parameters(), lr=0.001)
        self.target = None
        self.stay_prob = 0.5

    def generate_dataset_flat(self, occupancy_map, terrain_map, s_bar_collection, a_bar_collection):

        total_len = len(a_bar_collection)

        occupancy_map_padded = np.ones((self.grid_dimensions[0]+2, self.grid_dimensions[1]+2))
        occupancy_map_padded[1:self.grid_dimensions[0]+1, 1:self.grid_dimensions[1]+1] = occupancy_map

        terrain_map_padded = np.zeros((self.grid_dimensions[0] + 2, self.grid_dimensions[1] + 2))
        terrain_map_padded[1:self.grid_dimensions[0] + 1, 1:self.grid_dimensions[1] + 1] = terrain_map

        x_arr = np.zeros((total_len, 19))
        y_arr = np.zeros((total_len, 1))
        w_arr = np.zeros((total_len, 9))

        for idx in range(total_len):
            s_bar = s_bar_collection[idx]
            x_arr[idx, :9] = occupancy_map_padded[s_bar[0]:s_bar[0] + 3, s_bar[1]:s_bar[1] + 3].flatten()
            x_arr[idx, 9:18] = terrain_map_padded[s_bar[0]:s_bar[0] + 3, s_bar[1]:s_bar[1] + 3].flatten()
            x_arr[idx, 18] = a_bar_collection[idx]

            s_new_bar = s_bar_collection[idx+1]
            state_diff_x = np.clip(round(s_new_bar[0] - s_bar[0]), -1, 1, dtype=np.int)
            state_diff_y = np.clip(round(s_new_bar[1] - s_bar[1]), -1, 1, dtype=np.int)
            if state_diff_x == 0 and state_diff_y == 1:
                diff_idx = 1
                w_arr[idx, 1] = 1
            elif state_diff_x == 1 and state_diff_y == 1:
                diff_idx = 2
                w_arr[idx, 2] = 1
            elif state_diff_x == 1 and state_diff_y == 0:
                diff_idx = 3
                w_arr[idx, 3] = 1
            elif state_diff_x == 1 and state_diff_y == -1:
                diff_idx = 4
                w_arr[idx, 4] = 1
            elif state_diff_x == 0 and state_diff_y == -1:
                diff_idx = 5
                w_arr[idx, 5] = 1
            elif state_diff_x == -1 and state_diff_y == -1:
                diff_idx = 6
                w_arr[idx, 6] = 1
            elif state_diff_x == -1 and state_diff_y == 0:
                diff_idx = 7
                w_arr[idx, 7] = 1
            elif state_diff_x == -1 and state_diff_y == 1:
                diff_idx = 8
                w_arr[idx, 8] = 1
            else:
                diff_idx = 0
                w_arr[idx, 0] = 1

            y_arr[idx] = diff_idx

        return x_arr, y_arr, w_arr

    def generate_dataset_flat_agents(self, occupancy_map_list, s_bar_collection, a_bar_collection):

        total_len = len(a_bar_collection)

        x_arr = np.zeros((total_len, 10))
        y_arr = np.zeros((total_len, 1))
        w_arr = np.zeros((total_len, 5))

        for idx in range(total_len):
            s_bar = s_bar_collection[idx]
            occupancy_map_padded = np.ones((self.grid_dimensions[0] + 2, self.grid_dimensions[1] + 2))
            occupancy_map_padded[1:self.grid_dimensions[0] + 1, 1:self.grid_dimensions[1] + 1] = occupancy_map_list[idx]
            x_arr[idx, :9] = occupancy_map_padded[s_bar[0]:s_bar[0] + 3, s_bar[1]:s_bar[1] + 3].flatten()
            x_arr[idx, 9] = a_bar_collection[idx]

            s_new_bar = s_bar_collection[idx+1]
            state_diff_x = round(s_new_bar[0] - s_bar[0])
            state_diff_y = round(s_new_bar[1] - s_bar[1])
            if state_diff_x == 0 and state_diff_y == 1:
                diff_idx = 1
                w_arr[idx, 1] = 1
            elif state_diff_x == 1 and state_diff_y == 0:
                diff_idx = 2
                w_arr[idx, 2] = 1
            elif state_diff_x == 0 and state_diff_y == -1:
                diff_idx = 3
                w_arr[idx, 3] = 1
            elif state_diff_x == -1 and state_diff_y == 0:
                diff_idx = 4
                w_arr[idx, 4] = 1
            else:
                diff_idx = 0
                w_arr[idx, 0] = 1

            y_arr[idx] = diff_idx

        return x_arr, y_arr, w_arr

    def train_net(self, buffer, bs, opt_iterations, rw=False):

        if rw:
            self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(buffer.get_weights()))
        else:
            self.criterion = nn.CrossEntropyLoss()

        for ijk in range(opt_iterations):
            x, y = buffer.sample(batch_size=bs)
            x_batch = torch.from_numpy(np.float32(x))
            y_batch = torch.from_numpy(np.int64(y))

            self.optimizer.zero_grad()

            outputs = self.P_net(x_batch)
            loss = self.criterion(outputs, y_batch.squeeze())
            loss.backward()
            self.optimizer.step()

    def update_p_table_optimistic(self, occupancy_map, walls=False):
        self.P_table[:, :, :, :] = 0.
        if walls:
            for idx_x in range(self.grid_dimensions[0]):
                for idx_y in range(self.grid_dimensions[1]):
                    for idx_a in range(8):
                        if idx_a == 0:
                            if (idx_y + 1 < self.grid_dimensions[1]) and (occupancy_map[idx_x, idx_y + 1] == 0):
                                self.P_table[idx_x, idx_y, 0, 1] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 0, 0] = 1.0
                        elif idx_a == 1:
                            if (idx_x + 1 < self.grid_dimensions[0]) and (idx_y + 1 < self.grid_dimensions[1]) and \
                                    (occupancy_map[idx_x + 1, idx_y + 1] == 0):
                                self.P_table[idx_x, idx_y, 1, 2] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 1, 0] = 1.0
                        elif idx_a == 2:
                            if (idx_x + 1 < self.grid_dimensions[0]) and (occupancy_map[idx_x + 1, idx_y] == 0):
                                self.P_table[idx_x, idx_y, 2, 3] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 2, 0] = 1.0
                        elif idx_a == 3:
                            if (idx_x + 1 < self.grid_dimensions[0]) and (idx_y - 1 > 0) and \
                                    (occupancy_map[idx_x + 1, idx_y - 1] == 0):
                                self.P_table[idx_x, idx_y, 3, 4] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 3, 0] = 1.0
                        elif idx_a == 4:
                            if (idx_y - 1 > 0) and (occupancy_map[idx_x, idx_y - 1] == 0):
                                self.P_table[idx_x, idx_y, 4, 5] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 4, 0] = 1.0
                        elif idx_a == 5:
                            if (idx_x - 1 > 0) and (idx_y - 1 > 0) and \
                                    (occupancy_map[idx_x - 1, idx_y - 1] == 0):
                                self.P_table[idx_x, idx_y, 5, 6] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 5, 0] = 1.0
                        elif idx_a == 6:
                            if (idx_x - 1 > 0) and (occupancy_map[idx_x - 1, idx_y] == 0):
                                self.P_table[idx_x, idx_y, 6, 7] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 6, 0] = 1.0
                        elif idx_a == 7:
                            if (idx_x - 1 > 0) and (idx_y + 1 < self.grid_dimensions[1]) and \
                                    (occupancy_map[idx_x - 1, idx_y + 1] == 0):
                                self.P_table[idx_x, idx_y, 7, 8] = 1.0
                            else:
                                self.P_table[idx_x, idx_y, 7, 0] = 1.0
        else:
            for idx_x in range(self.grid_dimensions[0]):
                for idx_y in range(self.grid_dimensions[1]):
                    for idx_a in range(8):
                        self.P_table[idx_x, idx_y, idx_a, idx_a + 1] = 1.0

    def update_stay_prob(self, stay_prob_new, alpha=0.1):
        self.stay_prob = alpha * stay_prob_new + (1 - alpha) * self.stay_prob

    def update_p_table(self, occupancy_map, terrain_map, walls=False):
        occupancy_map_padded = np.ones((self.grid_dimensions[0] + 2, self.grid_dimensions[1] + 2))
        occupancy_map_padded[1:self.grid_dimensions[0] + 1, 1:self.grid_dimensions[1] + 1] = occupancy_map

        terrain_map_padded = np.zeros((self.grid_dimensions[0] + 2, self.grid_dimensions[1] + 2))
        terrain_map_padded[1:self.grid_dimensions[0] + 1, 1:self.grid_dimensions[1] + 1] = terrain_map

        for idx_x in range(self.grid_dimensions[0]):
            for idx_y in range(self.grid_dimensions[1]):
                for idx_a in range(8):
                    x = np.zeros((19,))
                    x[:9] = occupancy_map_padded[idx_x:idx_x + 3, idx_y:idx_y + 3].flatten()
                    x[9:18] = terrain_map_padded[idx_x:idx_x + 3, idx_y:idx_y + 3].flatten()
                    x[18] = idx_a
                    softmax = torch.nn.Softmax(dim=1)
                    self.P_table[idx_x, idx_y, idx_a, :] = softmax(
                        self.P_net(torch.from_numpy(np.float32(np.expand_dims(x, axis=0))))).data.numpy()

                if walls and (occupancy_map[idx_x, idx_y] == 1):
                    self.P_table[idx_x, idx_y, :, :] = 0

    def run_vi(self, grid, goal, num_runs=100, V_init=None):

        if V_init is not None:
            V = copy.deepcopy(V_init)
            V_old = copy.deepcopy(V_init)
        else:
            V = np.zeros(self.grid_dimensions)
            V_old = np.zeros(self.grid_dimensions)
        PI = np.zeros(self.grid_dimensions)

        R = -1 * np.ones(self.grid_dimensions)
        R[goal[0], goal[1]] = 0.
        R_tensor = np.zeros((grid.x_size + 2, grid.y_size + 2, 9))
        rows, columns, _ = R_tensor.shape
        R_tensor[1:rows - 1, 1:columns - 1, 0] = R
        R_tensor[1:rows - 1, 0:columns - 2, 1] = R
        R_tensor[0:rows - 2, 0:columns - 2, 2] = R
        R_tensor[0:rows - 2, 1:columns - 1, 3] = R
        R_tensor[0:rows - 2, 2:columns, 4] = R
        R_tensor[1:rows - 1, 2:columns, 5] = R
        R_tensor[2:rows, 2:columns, 6] = R
        R_tensor[2:rows, 1:columns - 1, 7] = R
        R_tensor[2:rows, 0:columns - 2, 8] = R
        R_tensor = R_tensor[1:rows - 1, 1:columns - 1, :]
        R_tensor = np.repeat(np.expand_dims(R_tensor, axis=2), 8, axis=2)

        P = copy.deepcopy(self.P_table)
        P[goal[0], goal[1], :, :] = np.zeros((8, 9))
        P[goal[0], goal[1], :, 0] = 1.0  # new

        mask = np.zeros((grid.x_size + 2, grid.y_size + 2, 9))
        mask[1:rows - 1, 1:columns - 1, 0] = grid.occupancy_map_un_padded
        mask[1:rows - 1, 0:columns - 2, 1] = grid.occupancy_map_un_padded
        mask[0:rows - 2, 0:columns - 2, 2] = grid.occupancy_map_un_padded
        mask[0:rows - 2, 1:columns - 1, 3] = grid.occupancy_map_un_padded
        mask[0:rows - 2, 2:columns, 4] = grid.occupancy_map_un_padded
        mask[1:rows - 1, 2:columns, 5] = grid.occupancy_map_un_padded
        mask[2:rows, 2:columns, 6] = grid.occupancy_map_un_padded
        mask[2:rows, 1:columns - 1, 7] = grid.occupancy_map_un_padded
        mask[2:rows, 0:columns - 2, 8] = grid.occupancy_map_un_padded
        mask = mask[1:rows - 1, 1:columns - 1, :]
        mask = np.repeat(np.expand_dims(mask, axis=2), 8, axis=2)

        vi_iter = 0

        for _ in range(num_runs):
            vi_iter += 1
            V_rep = np.zeros((grid.x_size + 2, grid.y_size + 2, 9))
            V_rep[1:rows - 1, 1:columns - 1, 0] = V
            V_rep[1:rows - 1, 0:columns - 2, 1] = V
            V_rep[0:rows - 2, 0:columns - 2, 2] = V
            V_rep[0:rows - 2, 1:columns - 1, 3] = V
            V_rep[0:rows - 2, 2:columns, 4] = V
            V_rep[1:rows - 1, 2:columns, 5] = V
            V_rep[2:rows, 2:columns, 6] = V
            V_rep[2:rows, 1:columns - 1, 7] = V
            V_rep[2:rows, 0:columns - 2, 8] = V
            V_rep = V_rep[1:rows - 1, 1:columns - 1, :]
            V_rep = np.repeat(np.expand_dims(V_rep, axis=2), 8, axis=2)

            V_prop = P * (R_tensor*mask + self.gamma*V_rep*mask)

            V_a = np.sum(V_prop, axis=3)

            V = np.max(V_a, axis=2)

            progress = (np.abs(V - V_old)).mean()
            V_old = copy.deepcopy(V)
            if progress < 1e-3:
                break

        for ix in range(grid.x_size):
            for iy in range(grid.y_size):

                if (np.max(V_a[ix, iy, :]) == 0) and (abs(V_a[ix, iy, :].mean() - V_a[ix, iy, 0]) < 1e-5):
                    PI[ix, iy] = randint(0, 7)
                else:
                    PI[ix, iy] = np.argmax(V_a[ix, iy, :])

        return V, PI

    def set_target(self, tile, action, mode='normal', occupancy_map=None):
        if mode == 'random':
            applicable = False
            while not applicable:
                action = random.randint(0, 7)
                if action == 0:
                    if (tile[1] + 1 < self.grid_dimensions[1]) and (occupancy_map[tile[0], tile[1] + 1] == 0):
                        applicable = True
                        self.target = (tile[0], tile[1] + 1)
                elif action == 1:
                    if (tile[1] + 1 < self.grid_dimensions[1]) and (tile[0] + 1 < self.grid_dimensions[0]) and \
                            (occupancy_map[tile[0] + 1, tile[1] + 1] == 0):
                        applicable = True
                        self.target = (tile[0] + 1, tile[1] + 1)
                elif action == 2:
                    if (tile[0] + 1 < self.grid_dimensions[0]) and (occupancy_map[tile[0] + 1, tile[1]] == 0):
                        applicable = True
                        self.target = (tile[0] + 1, tile[1])
                elif action == 3:
                    if (tile[0] + 1 < self.grid_dimensions[0]) and (tile[1] - 1 > 0) and \
                            (occupancy_map[tile[0] + 1, tile[1] - 1] == 0):
                        applicable = True
                        self.target = (tile[0] + 1, tile[1] - 1)
                elif action == 4:
                    if (tile[1] - 1 > 0) and (occupancy_map[tile[0], tile[1] - 1] == 0):
                        applicable = True
                        self.target = (tile[0], tile[1] - 1)
                elif action == 5:
                    if (tile[0] - 1 > 0) and (tile[1] - 1 > 0) and (occupancy_map[tile[0] - 1, tile[1] - 1] == 0):
                        applicable = True
                        self.target = (tile[0] - 1, tile[1] - 1)
                elif action == 6:
                    if (tile[0] - 1 > 0) and (occupancy_map[tile[0] - 1, tile[1]] == 0):
                        applicable = True
                        self.target = (tile[0] - 1, tile[1])
                elif action == 7:
                    if (tile[1] + 1 < self.grid_dimensions[1]) and (tile[0] - 1 > 0) and \
                            (occupancy_map[tile[0] - 1, tile[1] + 1] == 0):
                        applicable = True
                        self.target = (tile[0] - 1, tile[1] + 1)
        else:
            if action == 0:
                self.target = (tile[0], tile[1] + 1)
            elif action == 1:
                self.target = (tile[0] + 1, tile[1] + 1)
            elif action == 2:
                self.target = (tile[0] + 1, tile[1])
            elif action == 3:
                self.target = (tile[0] + 1, tile[1] - 1)
            elif action == 4:
                self.target = (tile[0], tile[1] - 1)
            elif action == 5:
                self.target = (tile[0] - 1, tile[1] - 1)
            elif action == 6:
                self.target = (tile[0] - 1, tile[1])
            elif action == 7:
                self.target = (tile[0] - 1, tile[1] + 1)
        self.target = (max(0, min(self.target[0], self.grid_dimensions[0] - 1)),
                       max(0, min(self.target[1], self.grid_dimensions[1] - 1)))

    def get_target_vec(self, state):
        return self.target[0] - state[0], self.target[1] - state[1]

    def get_target_vec_t2(self, state):
        return 0.5*self.target[0] - state[0], 0.5*self.target[1] - state[1]

    def get_target_vec_mj_point(self, state):
        return 0.25*self.target[0] - 0.875 - state[0], 0.25*self.target[1] - 0.875 - state[1]

    def get_target_vec_mj_ant(self, state):
        return self.target[0] - 1.0 - state[0], self.target[1] - 1.0 - state[1]

    def get_target(self):
        return self.target
