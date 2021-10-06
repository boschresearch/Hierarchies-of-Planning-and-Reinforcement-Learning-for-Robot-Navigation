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
import random

import torch
import torch.nn as nn
import torch.optim as optim

from lib.networks.transition import TransNetFlatParkingO


class ValueIteration33SyncNN:

    def __init__(self, grid_size_x, grid_size_y, orient_res=8,  gamma=0.99):
        self.grid_dimensions = (grid_size_x, grid_size_y, orient_res)
        self.gamma = gamma
        self.P_net = TransNetFlatParkingO()
        self.P_table = np.zeros((grid_size_x, grid_size_y, orient_res, 8, orient_res, 9, orient_res))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.P_net.parameters(), lr=0.001)
        self.target = None
        self.stay_prob = 0.5

    def generate_dataset_flat(self, s_bar_collection, a_bar_collection, s_new_bar_collection):

        total_len = len(s_bar_collection)

        x_arr = np.zeros((total_len, 3))
        y_arr = np.zeros((total_len, 2))

        for idx in range(total_len):
            s_bar = s_bar_collection[idx]
            x_arr[idx, 0] = s_bar_collection[idx][2]
            x_arr[idx, 1] = a_bar_collection[idx][0]
            x_arr[idx, 2] = a_bar_collection[idx][1]

            s_new_bar = s_new_bar_collection[idx]
            state_diff_x = np.clip(round(s_new_bar[0] - s_bar[0]), -1, 1)
            state_diff_y = np.clip(round(s_new_bar[1] - s_bar[1]), -1, 1)
            if state_diff_x == 0 and state_diff_y == 1:
                diff_idx = 1
            elif state_diff_x == 1 and state_diff_y == 1:
                diff_idx = 2
            elif state_diff_x == 1 and state_diff_y == 0:
                diff_idx = 3
            elif state_diff_x == 1 and state_diff_y == -1:
                diff_idx = 4
            elif state_diff_x == 0 and state_diff_y == -1:
                diff_idx = 5
            elif state_diff_x == -1 and state_diff_y == -1:
                diff_idx = 6
            elif state_diff_x == -1 and state_diff_y == 0:
                diff_idx = 7
            elif state_diff_x == -1 and state_diff_y == 1:
                diff_idx = 8
            else:
                diff_idx = 0

            y_arr[idx, 0] = diff_idx
            y_arr[idx, 1] = s_new_bar[2]

        return x_arr, y_arr

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

            out_p, out_o = self.P_net(x_batch)
            loss_p = self.criterion(out_p, y_batch[:, 0].squeeze())
            loss_o = self.criterion(out_o, y_batch[:, 1].squeeze())
            loss = loss_p + loss_o
            loss.backward()
            self.optimizer.step()

    def update_p_table_optimistic(self, occupancy_map, walls=False):
        self.P_table[:, :, :, :, :, :, :] = 0.
        if walls:
            for idx_x in range(self.grid_dimensions[0]):
                for idx_y in range(self.grid_dimensions[1]):
                    for idx_o in range(self.grid_dimensions[2]):
                        for idx_ao in range(self.grid_dimensions[2]):
                            for idx_a in range(8):
                                if idx_a == 0:
                                    if (idx_y + 1 < self.grid_dimensions[1]) and (occupancy_map[idx_x, idx_y + 1] == 0):
                                        self.P_table[idx_x, idx_y, idx_o, 0, idx_ao, 1, idx_ao] = 1.0
                                    else:
                                        self.P_table[idx_x, idx_y, idx_o, 0, idx_ao, 0, idx_ao] = 1.0
                                elif idx_a == 1:
                                    if (idx_x + 1 < self.grid_dimensions[0]) and (
                                            idx_y + 1 < self.grid_dimensions[1]) and \
                                            (occupancy_map[idx_x + 1, idx_y + 1] == 0):
                                        self.P_table[idx_x, idx_y, idx_o, 1, idx_ao, 2, idx_ao] = 1.0
                                    else:
                                        self.P_table[idx_x, idx_y, idx_o, 1, idx_ao, 0, idx_ao] = 1.0
                                elif idx_a == 2:
                                    if (idx_x + 1 < self.grid_dimensions[0]) and (occupancy_map[idx_x + 1, idx_y] == 0):
                                        self.P_table[idx_x, idx_y, idx_o, 2, idx_ao, 3, idx_ao] = 1.0
                                    else:
                                        self.P_table[idx_x, idx_y, idx_o, 2, idx_ao, 0, idx_ao] = 1.0
                                elif idx_a == 3:
                                    if (idx_x + 1 < self.grid_dimensions[0]) and (idx_y - 1 > 0) and \
                                            (occupancy_map[idx_x + 1, idx_y - 1] == 0):
                                        self.P_table[idx_x, idx_y, idx_o, 3, idx_ao, 4, idx_ao] = 1.0
                                    else:
                                        self.P_table[idx_x, idx_y, idx_o, 3, idx_ao, 0, idx_ao] = 1.0
                                elif idx_a == 4:
                                    if (idx_y - 1 > 0) and (occupancy_map[idx_x, idx_y - 1] == 0):
                                        self.P_table[idx_x, idx_y, idx_o, 4, idx_ao, 5, idx_ao] = 1.0
                                    else:
                                        self.P_table[idx_x, idx_y, idx_o, 4, idx_ao, 0, idx_ao] = 1.0
                                elif idx_a == 5:
                                    if (idx_x - 1 > 0) and (idx_y - 1 > 0) and \
                                            (occupancy_map[idx_x - 1, idx_y - 1] == 0):
                                        self.P_table[idx_x, idx_y, idx_o, 5, idx_ao, 6, idx_ao] = 1.0
                                    else:
                                        self.P_table[idx_x, idx_y, idx_o, 5, idx_ao, 0, idx_ao] = 1.0
                                elif idx_a == 6:
                                    if (idx_x - 1 > 0) and (occupancy_map[idx_x - 1, idx_y] == 0):
                                        self.P_table[idx_x, idx_y, idx_o, 6, idx_ao, 7, idx_ao] = 1.0
                                    else:
                                        self.P_table[idx_x, idx_y, idx_o, 6, idx_ao, 0, idx_ao] = 1.0
                                elif idx_a == 7:
                                    if (idx_x - 1 > 0) and (idx_y + 1 < self.grid_dimensions[1]) and \
                                            (occupancy_map[idx_x - 1, idx_y + 1] == 0):
                                        self.P_table[idx_x, idx_y, idx_o, 7, idx_ao, 8, idx_ao] = 1.0
                                    else:
                                        self.P_table[idx_x, idx_y, idx_o, 7, idx_ao, 0, idx_ao] = 1.0
        else:
            for idx_x in range(self.grid_dimensions[0]):
                for idx_y in range(self.grid_dimensions[1]):
                    for idx_o in range(self.grid_dimensions[2]):
                        for idx_ao in range(self.grid_dimensions[2]):
                            for idx_a in range(8):
                                self.P_table[idx_x, idx_y, idx_o, idx_a, idx_ao, idx_a + 1, idx_ao] = 1.0

    def update_p_table(self, occupancy_map, walls=False):
        occupancy_map_padded = np.ones((self.grid_dimensions[0] + 2, self.grid_dimensions[1] + 2))
        occupancy_map_padded[1:self.grid_dimensions[0] + 1, 1:self.grid_dimensions[1] + 1] = occupancy_map

        for idx_o in range(self.grid_dimensions[2]):
            for idx_a in range(8):
                for idx_ao in range(self.grid_dimensions[2]):
                    x = np.array([idx_o, idx_a, idx_ao])
                    softmax = torch.nn.Softmax(dim=1)
                    p_p, p_o = self.P_net(torch.from_numpy(np.float32(np.expand_dims(x, axis=0))))
                    prob_p = softmax(p_p).detach()
                    prob_o = softmax(p_o).detach()
                    for n_p in range(9):
                        for n_o in range(self.grid_dimensions[2]):
                            prob = prob_p[0, n_p] * prob_o[0, n_o]
                            self.P_table[:, :, idx_o, idx_a, idx_ao, n_p, n_o] = prob

    def run_vi(self, grid, goal, num_runs=100):

        V = np.zeros(self.grid_dimensions)
        V_old = np.zeros(self.grid_dimensions)
        PI = np.zeros((self.grid_dimensions[0], self.grid_dimensions[1], self.grid_dimensions[2], 2))

        R = -1 * np.ones(self.grid_dimensions)
        R[goal[0], goal[1], goal[2]] = 0.
        R_tensor = np.zeros((grid.x_size + 2, grid.y_size + 2, self.grid_dimensions[2], 9))
        rows, columns, _, _ = R_tensor.shape

        P = copy.deepcopy(self.P_table)
        P[goal[0], goal[1], goal[2], :, :, :, :] = 0.0
        P[goal[0], goal[1], goal[2], :, :, 0, :] = 1.0  # new

        for _ in range(num_runs):
            V[goal[0], goal[1], goal[2]] = 1.0 / self.gamma  # new
            V_rep = np.zeros((grid.x_size + 2, grid.y_size + 2, self.grid_dimensions[2], 9))
            V_rep[1:rows - 1, 1:columns - 1, :, 0] = V
            V_rep[1:rows - 1, 0:columns - 2, :, 1] = V
            V_rep[0:rows - 2, 0:columns - 2, :, 2] = V
            V_rep[0:rows - 2, 1:columns - 1, :, 3] = V
            V_rep[0:rows - 2, 2:columns, :, 4] = V
            V_rep[1:rows - 1, 2:columns, :, 5] = V
            V_rep[2:rows, 2:columns, :, 6] = V
            V_rep[2:rows, 1:columns - 1, :, 7] = V
            V_rep[2:rows, 0:columns - 2, :, 8] = V
            V_rep = V_rep[1:rows - 1, 1:columns - 1, :, :]
            V_rep = np.repeat(np.expand_dims(V_rep, axis=3), 8, axis=3)

            V_rep_2 = np.zeros((grid.x_size, grid.y_size, self.grid_dimensions[2], 8, 9, 8))
            V_rep_2[:, :, :, :, :, 0] = np.repeat(np.expand_dims(V_rep[:, :, 0, :, :], axis=3), 8, axis=3)
            V_rep_2[:, :, :, :, :, 1] = np.repeat(np.expand_dims(V_rep[:, :, 1, :, :], axis=3), 8, axis=3)
            V_rep_2[:, :, :, :, :, 2] = np.repeat(np.expand_dims(V_rep[:, :, 2, :, :], axis=3), 8, axis=3)
            V_rep_2[:, :, :, :, :, 3] = np.repeat(np.expand_dims(V_rep[:, :, 3, :, :], axis=3), 8, axis=3)
            V_rep_2[:, :, :, :, :, 4] = np.repeat(np.expand_dims(V_rep[:, :, 4, :, :], axis=3), 8, axis=3)
            V_rep_2[:, :, :, :, :, 5] = np.repeat(np.expand_dims(V_rep[:, :, 5, :, :], axis=3), 8, axis=3)
            V_rep_2[:, :, :, :, :, 6] = np.repeat(np.expand_dims(V_rep[:, :, 6, :, :], axis=3), 8, axis=3)
            V_rep_2[:, :, :, :, :, 7] = np.repeat(np.expand_dims(V_rep[:, :, 7, :, :], axis=3), 8, axis=3)
            V_rep_2 = np.repeat(np.expand_dims(V_rep_2, axis=4), 8, axis=4)

            V_prop = P * self.gamma * V_rep_2

            V_a = np.sum(V_prop, axis=6)
            V_a = np.sum(V_a, axis=5)

            V = np.max(V_a, axis=4)
            V = np.max(V, axis=3)

            progress = (np.abs(V - V_old)).mean()
            V_old = copy.deepcopy(V)
            if progress < 1e-3:
                break

        for ix in range(grid.x_size):
            for iy in range(grid.y_size):
                for io in range(self.grid_dimensions[2]):
                    x_ind, y_ind = np.where(V_a[ix, iy, io, :, :] == V_a[ix, iy, io, :, :].max())
                    selected = np.random.randint(x_ind.shape[0])
                    PI[ix, iy, io, 0] = x_ind[selected]
                    PI[ix, iy, io, 1] = y_ind[selected]

        return V, PI

    def set_target(self, tile, action, mode='normal', occupancy_map=None, keep_goal=False):
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
            if keep_goal:
                self.target = (tile[0], tile[1], action[1])
            else:
                if action[0] == 0:
                    self.target = (tile[0], tile[1] + 1, action[1])
                elif action[0] == 1:
                    self.target = (tile[0] + 1, tile[1] + 1, action[1])
                elif action[0] == 2:
                    self.target = (tile[0] + 1, tile[1], action[1])
                elif action[0] == 3:
                    self.target = (tile[0] + 1, tile[1] - 1, action[1])
                elif action[0] == 4:
                    self.target = (tile[0], tile[1] - 1, action[1])
                elif action[0] == 5:
                    self.target = (tile[0] - 1, tile[1] - 1, action[1])
                elif action[0] == 6:
                    self.target = (tile[0] - 1, tile[1], action[1])
                elif action[0] == 7:
                    self.target = (tile[0] - 1, tile[1] + 1, action[1])
        self.target = (max(0, min(self.target[0], self.grid_dimensions[0] - 1)),
                       max(0, min(self.target[1], self.grid_dimensions[1] - 1)),
                       self.target[2])

    def get_target_vec(self, state):
        if self.target[2] == 0:
            angle = 0
        elif self.target[2] == 1:
            angle = np.pi/4
        elif self.target[2] == 2:
            angle = np.pi/2
        elif self.target[2] == 3:
            angle = 3*np.pi/4
        elif self.target[2] == 4:
            angle = np.pi
        elif self.target[2] == 5:
            angle = -3*np.pi/4
        elif self.target[2] == 6:
            angle = -np.pi/2
        elif self.target[2] == 7:
            angle = -np.pi/4
        ch = np.cos(angle)
        sh = np.sin(angle)
        return (self.target[0] / 25) - 0.46 - state[0], (self.target[1] / 25) - 0.22 - state[1], ch - state[4], \
               sh - state[5]

    def get_target(self):
        return self.target
