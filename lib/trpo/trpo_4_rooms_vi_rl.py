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

import torch
import torch.nn.functional as F
import numpy as np

from random import randint

from lib.policy.batch import Batch
from lib.policy.continuous_mlp import ContinuousMLP
from lib.optimizers.actor_critic.actor_critic import TRPO

from lib.vi.value_iteration_new import ValueIteration4M1SyncNN8
from lib.environments.gridworld import GridWorldContinuous


class Buffer:
    def __init__(self, max_size=int(16000)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.x = np.zeros((max_size, 10))
        self.y = np.zeros((max_size, 1))
        self.w = np.zeros((max_size, 9))

    def add(self, x, y, w):
        self.x[self.ptr] = x
        self.y[self.ptr] = y
        self.w[self.ptr] = w

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
                 env_mode,
                 op_mode,
                 state_dim,
                 action_dim,
                 config,
                 save_path=None
                 ):
        self.config = config
        self.save_path = save_path
        self.grid = GridWorldContinuous(grid_mode='custom',
                                        layout=np.load('maze1_layout.npy'),
                                        objects=np.load('maze1_objects.npy'),
                                        terrain=self.config.terrain_var)
        self.grid.goal = (13., 13.)
        self.grid.reset_env_terrain()

        self.policy = ContinuousMLP(state_dim,
                                    action_dim,
                                    hidden_sizes=config.policy_hidden_sizes,
                                    activation=F.relu,
                                    is_disc_action=False)
        self.optimizer = TRPO(policy=self.policy,
                              use_gpu=False,
                              max_kl=config.policy_max_kl,
                              damping=config.policy_damp_val,
                              use_fim=False,
                              discount=config.discount_factor,
                              imp_weight=False)
        self.buffer = Buffer()
        self.vi = ValueIteration4M1SyncNN8(grid_size_x=21,
                                           grid_size_y=21,
                                           gamma=self.optimizer.discount)

        self.env_mode = env_mode
        self.op_mode = op_mode
        self.state_dim = state_dim
        self.dqn_steps = 0
        self.eps = 1.0
        self.time_scale = 2
        self.episode_steps = 0

    def bound_x(self, input_var):
        return max(0, min(input_var, self.grid.x_size - 1))

    def bound_y(self, input_var):
        return max(0, min(input_var, self.grid.y_size - 1))

    def roll_out_in_env(self, start, goal, ultimate_goal, horizon, mode='test'):
        s_u_goal = self.grid.lls2hls(ultimate_goal)
        roll_out = Batch()
        break_var = False
        s = start
        d = False

        s_list = []
        a_list = []
        r_list = []

        for step_i in range(horizon):
            self.episode_steps += 1
            s_bar = self.grid.lls2hls(s)

            # GET THE TARGET VECTOR
            target_vec = (goal[0] - s[0], goal[1] - s[1])
            s_save = copy.deepcopy(np.array(list(s[2:]) + list(self.grid.state_cache[s_bar[0], s_bar[1]]) +
                                            list(target_vec)))
            s_pos_save = copy.deepcopy(np.array(s[:2]))
            s_list.append(np.concatenate((s_pos_save, s_save)))
            s = torch.tensor(s_save, dtype=torch.float).unsqueeze(0)
            a = self.policy.select_action(s)

            s_new, r, d, _, info = self.grid.step(action=10*a)
            s_bar_cand = self.grid.lls2hls(s_new)

            d = (s_bar_cand[0] == goal[0]) and (s_bar_cand[1] == goal[1])
            if d:
                info = False
                r = 1.0

            success_var = ((s_bar_cand[0] == s_u_goal[0]) and (s_bar_cand[1] == s_u_goal[1]))
            if success_var:
                info = False
                break_var = True

            break_var = break_var or (not info) or (step_i + 1 == horizon) or (self.episode_steps == self.config.time_horizon)

            a_list.append(a)
            r_list.append(r)
            roll_out.append(
                Batch([a.astype(np.float32)],
                      [s_save.astype(np.float32)],
                      [r],
                      [s_new.astype(np.float32)],
                      [0 if break_var else 1],
                      [info],
                      [1.0]))
            s = s_new
            if break_var:
                break

        return roll_out, s_list, a_list, r_list, d, s_new, s_bar_cand

    def simulate_env(self, mode):
        batch = Batch()
        num_roll_outs = 0
        num_steps = 0
        total_success = 0
        total_wp_success = 0
        j = 0.
        jwp = 0.

        if mode == 'train':

            while num_steps < self.config.policy_batch_size:

                """ INITIALIZE THE ENVIRONMENT """
                self.grid.reset_env_terrain()
                start_pos = self.grid.sample_random_start_terrain(number=1)[0]
                goal_pos = self.grid.sample_random_goal_terrain(number=1)[0]
                s_goal = self.grid.lls2hls(goal_pos)
                s_init = self.grid.reset(start_pos, goal_pos)
                self.episode_steps = 0

                """ V MAP """
                if self.config.optimistic_model:
                    self.vi.update_p_table_optimistic(occupancy_map=self.grid.occupancy_map, walls=True)
                else:
                    self.vi.update_p_table(occupancy_map=self.grid.terrain_map, walls=True)

                v, pi = self.vi.run_vi(grid=self.grid, goal=(s_goal[0], s_goal[1]))

                """ START THE EPISODE """
                horizon_left = self.config.time_horizon
                st = start_pos
                success = False

                s_bar = self.grid.lls2hls(st)
                hl_s_list = []
                hl_a_list = []
                hl_r_list = []
                hl_d_list = []
                hl_s_list.append(s_bar)

                while (horizon_left > 0) and not success:

                    # GET THE TARGET VECTOR
                    self.dqn_steps += 1
                    self.eps = 0.01 + 0.99 * math.exp(-1. * self.dqn_steps / 10000)

                    s_bar = self.grid.lls2hls(st)

                    if torch.rand(1)[0] > self.eps:
                        a_bar = int(pi[s_bar[0], s_bar[1]])
                    else:
                        a_bar = randint(0, 7)

                    self.vi.set_target(s_bar, a_bar)
                    curr_goal = self.vi.get_target()

                    roll_out, _, _, _, wp_success, l_state, s_bar_p = self.roll_out_in_env(
                        start=s_init,
                        goal=curr_goal,
                        horizon=self.time_scale,
                        ultimate_goal=goal_pos,
                        mode='train'
                    )

                    hl_s_list.append(s_bar_p)
                    hl_a_list.append(a_bar)

                    st = l_state[:2]
                    s_init = l_state

                    num_roll_outs += 1
                    num_steps += roll_out.length()
                    horizon_left -= roll_out.length()

                    total_wp_success += wp_success
                    jwp += 1

                    st_bar = self.grid.lls2hls(l_state)
                    success = ((st_bar[0] == s_goal[0]) and (st_bar[1] == s_goal[1]))
                    if success:
                        hl_r_list.append(0)
                        hl_d_list.append(True)
                    else:
                        hl_r_list.append(-1)
                        hl_d_list.append(False)

                    batch.append(roll_out)

                total_success += success
                j += 1

                if not self.config.optimistic_model:
                    x_temp, y_temp, w_temp = self.vi.generate_dataset_flat(self.grid.terrain_map,
                                                                           hl_s_list,
                                                                           hl_a_list
                                                                           )

                    for bi in range(x_temp.shape[0]):
                        self.buffer.add(x_temp[bi], y_temp[bi], w_temp[bi])

                    self.vi.train_net(buffer=self.buffer, bs=128, opt_iterations=40, rw=True)

            return batch, total_success / j, total_wp_success / jwp, num_steps / j, num_steps / num_roll_outs

        else:
            self.grid.reset_env_terrain()
            start_pos = self.grid.sample_random_start_terrain(number=1)[0]
            goal_pos = self.grid.sample_random_goal_terrain(number=1)[0]
            s_goal = self.grid.lls2hls(goal_pos)
            s_init = self.grid.reset(start_pos, goal_pos)
            self.episode_steps = 0

            """ V MAP """
            if self.config.optimistic_model:
                self.vi.update_p_table_optimistic(occupancy_map=self.grid.occupancy_map, walls=True)
            else:
                self.vi.update_p_table(occupancy_map=self.grid.terrain_map, walls=True)

            v, pi = self.vi.run_vi(grid=self.grid, goal=(s_goal[0], s_goal[1]))

            horizon_left = self.config.time_horizon
            st = start_pos
            success = False

            s_bar = self.grid.lls2hls(st)

            while (horizon_left > 0) and not success:

                # GET THE TARGET VECTOR
                a_bar = int(pi[s_bar[0], s_bar[1]])
                self.vi.set_target(s_bar, a_bar)
                curr_goal = self.vi.get_target()

                roll_out, states, actions, rewards, wp_success, l_state, _ = self.roll_out_in_env(
                    start=s_init,
                    goal=curr_goal,
                    horizon=self.time_scale,
                    ultimate_goal=goal_pos,
                    mode='test'
                )

                st = l_state[:2]
                s_bar = self.grid.lls2hls(st)
                s_init = l_state

                num_roll_outs += 1
                num_steps += roll_out.length()
                horizon_left -= roll_out.length()

                total_wp_success += wp_success
                jwp += 1

                st_bar = self.grid.lls2hls(l_state)
                success = ((st_bar[0] == s_goal[0]) and (st_bar[1] == s_goal[1]))

            return success

    def train(self):
        for iter_loop in range(self.config.num_inner_episodes):
            batch, g_success, wp_success, steps_trajectory, steps_wp = self.simulate_env(mode='train')
            self.optimizer.process_batch(self.policy, batch, [])
            print("G_S, WP_S, G_ST, WP_ST")
            print(g_success)
            print(wp_success)
            print(steps_trajectory)
            print(steps_wp)
            print(self.buffer.size)
            print(self.eps)

    def test(self):
        g_success = self.simulate_env(mode='test')
        return g_success
