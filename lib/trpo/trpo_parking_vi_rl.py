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
import math
import numpy as np
import torch
import torch.nn.functional as F
import gym
import highway_env

from random import randint

from lib.policy.batch import Batch
from lib.policy.continuous_mlp import ContinuousMLP
from lib.optimizers.actor_critic.actor_critic import TRPO

from lib.environments.parking import ParkingEmpty

from lib.vi.value_iteration_parking_orientation_new import ValueIteration33SyncNN


class Buffer:
    def __init__(self, max_size=int(16000)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.x = np.zeros((max_size, 3))
        self.y = np.zeros((max_size, 2))
        self.w = np.zeros((max_size, 5))

    def add(self, x, y):
        self.x[self.ptr] = x
        self.y[self.ptr] = y
        # self.w[self.ptr] = w

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            self.x[ind],
            self.y[ind]
        )

    def get_weights(self):
        class_sum = np.sum(self.w, axis=0)
        class_sum = class_sum / sum(class_sum)
        class_sum += 1e-3 * np.ones_like(class_sum)
        class_sum = np.array([1 / e for e in list(class_sum)])
        return class_sum / sum(class_sum)


class TRPOTrainer:

    def __init__(self,
                 op_mode,
                 state_dim,
                 action_dim,
                 hidden_sizes,
                 max_kl,
                 damping,
                 batch_size,
                 inner_episodes,
                 max_iter,
                 optimistic_model,
                 use_fim=False,
                 use_gpu=False
                 ):

        self.grid = ParkingEmpty()
        self.policy = ContinuousMLP(state_dim,
                                    action_dim,
                                    hidden_sizes=hidden_sizes,
                                    activation=F.relu,
                                    is_disc_action=False)
        self.optimizer = TRPO(policy=self.policy,
                              use_gpu=use_gpu,
                              max_kl=max_kl,
                              damping=damping,
                              use_fim=use_fim,
                              discount=0.99,
                              imp_weight=False)
        self.buffer = Buffer()
        gx = self.grid.x_size
        gy = self.grid.y_size
        self.vi = ValueIteration33SyncNN(grid_size_x=gx,
                                         grid_size_y=gy,
                                         gamma=self.optimizer.discount)

        self.op_mode = op_mode
        self.optimistic_model = optimistic_model
        self.batch_size = batch_size
        self.inner_episodes = inner_episodes
        self.max_iter = max_iter
        self.state_dim = state_dim
        self.dqn_steps = 0
        self.eps = 1.0
        self.time_scale = 2
        self.episode_steps = 0

    def roll_out_in_env(self, start, horizon):
        roll_out = Batch()
        s = start
        s_list = []
        a_list = []
        r_list = []

        d = False
        oob = False
        break_var = False
        s_u_goal = self.grid.lls2hls(s['observation'])
        target_vec = self.vi.get_target_vec(s['observation'])
        goal = self.grid.lls2hls(s['observation'] + np.array(list(target_vec[:2]) + [0, 0] + list(target_vec[2:4])))

        for step_i in range(horizon):
            self.episode_steps += 1

            target_vec = self.vi.get_target_vec(s['observation'])
            s_save = copy.deepcopy(np.array(list(target_vec[:2]) + list(s['observation'][2:4]) + list(target_vec[2:4])))

            s_list.append(copy.deepcopy(s['observation']))
            s_tensor = torch.tensor(s_save, dtype=torch.float).unsqueeze(0)
            a = self.policy.select_action(s_tensor)

            s_new, r, d, info = self.grid.env.step(a)
            s_new_bar = self.grid.lls2hls(s_new['observation'])
            success_var = info["is_success"]
            info = not info["is_success"]

            d = (s_new_bar[0] == goal[0]) and (s_new_bar[1] == goal[1]) and (s_new_bar[2] == goal[2])
            r = 0.0
            if d:
                r = 1.0
                info = False

            if success_var:
                info = False
                break_var = True

            break_var = break_var or (not info) or (step_i + 1 == horizon) or (self.episode_steps == self.max_iter)

            ib = self.grid.check_in_bounds(s_new['observation'])
            if not ib:
                oob = True

            a_list.append(a)
            r_list.append(r)
            roll_out.append(
                Batch([a.astype(np.float32)],
                      [s_save.astype(np.float32)],
                      [r],
                      [s_new['observation'].astype(np.float32)],
                      [0 if (break_var or oob) else 1],
                      [info],
                      [1.0]))

            s = s_new
            if break_var or oob:
                break

        s_list.append(copy.deepcopy(s['observation']))

        return roll_out, s_list, a_list, r_list, d, success_var, s_new, s_new_bar, oob

    def simulate_env(self, mode):
        batch = Batch()
        num_roll_outs = 0
        num_steps = 0
        total_success = 0
        total_wp_success = 0
        j = 0.
        jwp = 0.

        if mode == 'train':

            while num_steps < self.batch_size:

                """ INITIALIZE THE ENVIRONMENT """
                s_init = self.grid.reset()
                s_start = self.grid.lls2hls(s_init['observation'])
                s_goal = self.grid.lls2hls(s_init['desired_goal'])
                self.episode_steps = 0

                """ V MAP """
                if self.optimistic_model:
                    self.vi.update_p_table_optimistic(occupancy_map=self.grid.occupancy_map, walls=False)
                else:
                    self.vi.update_p_table(occupancy_map=self.grid.occupancy_map, walls=True)

                v, pi = self.vi.run_vi(grid=self.grid, goal=(s_goal[0], s_goal[1], s_goal[2]))

                """ START THE EPISODE """
                horizon_left = self.max_iter
                success = False

                hl_s_list = []
                hl_s_new_list = []
                hl_a_list = []
                hl_r_list = []
                hl_d_list = []

                while (horizon_left > 0) and not success:

                    # GET THE TARGET VECTOR
                    self.dqn_steps += 1
                    self.eps = 0.01 + 0.99 * math.exp(-1. * self.dqn_steps / 10000)

                    s_bar = self.grid.lls2hls(s_init['observation'])
                    hl_s_list.append(s_bar)

                    if torch.rand(1)[0] > self.eps:
                        a_bar = pi[s_bar[0], s_bar[1], s_bar[2]]
                    else:
                        a_bar = (randint(0, 7), randint(0, 7))

                    self.vi.set_target(s_bar, a_bar)

                    roll_out, _, _, _, wp_success, success, l_state, s_bar_p, oob = self.roll_out_in_env(
                        horizon=self.time_scale,
                        start=s_init)

                    hl_s_new_list.append(s_bar_p)
                    hl_a_list.append(a_bar)

                    s_init = l_state

                    num_roll_outs += 1
                    num_steps += roll_out.length()
                    horizon_left -= roll_out.length()

                    total_wp_success += wp_success
                    jwp += 1

                    if success:
                        hl_r_list.append(0)
                        hl_d_list.append(True)
                    else:
                        hl_r_list.append(-1)
                        hl_d_list.append(False)

                    batch.append(roll_out)

                    if oob:
                        break

                total_success += success
                j += 1

                if not self.optimistic_model:
                    x_temp, y_temp = self.vi.generate_dataset_flat(hl_s_list,
                                                                   hl_a_list,
                                                                   hl_s_new_list
                                                                   )

                    for bi in range(x_temp.shape[0]):
                        self.buffer.add(x_temp[bi], y_temp[bi])

                    self.vi.train_net(buffer=self.buffer, bs=128, opt_iterations=40)

            return batch, total_success / j, total_wp_success / jwp, num_steps / j, num_steps / num_roll_outs

        else:

            """ INITIALIZE THE ENVIRONMENT """
            s_init = self.grid.reset()
            s_start = self.grid.lls2hls(s_init['observation'])
            s_goal = self.grid.lls2hls(s_init['desired_goal'])
            self.episode_steps = 0

            """ V MAP """
            if self.optimistic_model:
                self.vi.update_p_table_optimistic(occupancy_map=self.grid.occupancy_map, walls=False)
            else:
                self.vi.update_p_table(occupancy_map=self.grid.occupancy_map, walls=True)

            v, pi = self.vi.run_vi(grid=self.grid, goal=(s_goal[0], s_goal[1], s_goal[2]))

            """ START THE EPISODE """
            horizon_left = self.max_iter
            success = False

            while (horizon_left > 0) and not success:

                # GET THE TARGET VECTOR
                s_bar = self.grid.lls2hls(s_init['observation'])
                a_bar = pi[s_bar[0], s_bar[1], s_bar[2]]

                self.vi.set_target(s_bar, a_bar)

                roll_out, states, actions, rewards, wp_success, success, l_state, s_bar_p, oob = self.roll_out_in_env(
                    horizon=self.time_scale,
                    start=s_init)

                s_init = l_state

                num_roll_outs += 1
                num_steps += roll_out.length()
                horizon_left -= roll_out.length()

                total_wp_success += wp_success
                jwp += 1

                if oob:
                    break

            return success

    def train(self):
        for iter_loop in range(self.inner_episodes):
            batch, g_success, wp_success, steps_trajectory, steps_wp = self.simulate_env(mode='train')
            self.optimizer.process_batch(self.policy, batch, [])
            print("G_S, WP_S, G_ST, WP_ST")
            print(g_success)
            print(wp_success)
            print(steps_trajectory)
            print(steps_wp)

    def test(self):
        g_success = self.simulate_env(mode='test')
        return g_success
