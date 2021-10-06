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

from lib.environments.gridworld import GridWorldContinuous

from lib.mvprop.mvprop import MVPROPFAT
from lib.mvprop.mvprop_optimizer import MVPROPOptimizerTD0
from lib.td3.utils import ReplayBufferVIN


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
        self.mvprop = MVPROPFAT(k=config.mvprop_k).cuda()
        self.mvprop_target = copy.deepcopy(self.mvprop).cuda()
        self.memory = ReplayBufferVIN(2, 1, self.grid.x_size, config.mvprop_buffer_size)
        self.mvprop_optimizer = MVPROPOptimizerTD0(self.mvprop,
                                                   self.mvprop_target,
                                                   self.memory,
                                                   config.discount_factor,
                                                   config.mvprop_batch_size,
                                                   config.mvprop_lr,
                                                   config.mvprop_k)

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

            s_new, r, d, d_wp, info = self.grid.step(action=10 * a)
            s_bar_cand = self.grid.lls2hls(s_new)

            d = (s_bar_cand[0] == goal[0]) and (s_bar_cand[1] == goal[1])
            if d:
                info = False
                r = 1.0

            success_var = ((s_bar_cand[0] == s_u_goal[0]) and (s_bar_cand[1] == s_u_goal[1]))
            if success_var:
                info = False
                break_var = True

            break_var = break_var or (not info) or (step_i + 1 == horizon) or (
                        self.episode_steps == self.config.time_horizon)

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

                """ IMAGE INPUT """
                image = np.zeros((1, 2, self.grid.x_size, self.grid.y_size))
                image[0, 0, :, :] = self.grid.terrain_map_un_padded
                image[0, 1, :, :] = -1 * np.ones((self.grid.x_size, self.grid.y_size))
                image[0, 1, s_goal[0], s_goal[1]] = 0
                image = torch.from_numpy(image).float().cuda()

                with torch.no_grad():
                    v = self.mvprop_optimizer.mvprop(image)
                    v = v.cpu().detach()

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
                    if self.config.mvprop_decay_type == 'exp':
                        self.eps = 0.01 + 0.99 * math.exp(-1. * self.dqn_steps / self.config.mvprop_decay)

                    if torch.rand(1)[0] > self.eps:
                        with torch.no_grad():
                            options_x = [s_bar[0], s_bar[0], s_bar[0] + 1, s_bar[0] + 1, s_bar[0] + 1, s_bar[0],
                                         s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1]
                            options_y = [s_bar[1], s_bar[1] + 1, s_bar[1] + 1, s_bar[1], s_bar[1] - 1, s_bar[1] - 1,
                                         s_bar[1] - 1, s_bar[1], s_bar[1] + 1]
                            options_x = [self.bound_x(e) for e in options_x]
                            options_y = [self.bound_y(e) for e in options_y]
                            v_options = v[0, 0, options_x, options_y]
                            option = np.argmax(v_options)
                    else:
                        option = randint(0, 8)

                    if option == 0:
                        target = (s_bar[0], s_bar[1])
                    elif option == 1:
                        target = (s_bar[0], s_bar[1] + 1)
                    elif option == 2:
                        target = (s_bar[0] + 1, s_bar[1] + 1)
                    elif option == 3:
                        target = (s_bar[0] + 1, s_bar[1])
                    elif option == 4:
                        target = (s_bar[0] + 1, s_bar[1] - 1)
                    elif option == 5:
                        target = (s_bar[0], s_bar[1] - 1)
                    elif option == 6:
                        target = (s_bar[0] - 1, s_bar[1] - 1)
                    elif option == 7:
                        target = (s_bar[0] - 1, s_bar[1])
                    elif option == 8:
                        target = (s_bar[0] - 1, s_bar[1] + 1)
                    target = (max(0, min(target[0], self.grid.x_size - 1)),
                              max(0, min(target[1], self.grid.y_size - 1)))

                    roll_out, _, _, _, wp_success, l_state, s_bar_p = self.roll_out_in_env(
                        start=s_init,
                        goal=target,
                        horizon=self.time_scale,
                        ultimate_goal=goal_pos,
                        mode='train'
                    )

                    s_init = l_state

                    hl_s_list.append(s_bar_p)
                    hl_a_list.append(option)

                    st = l_state[:2]
                    s_bar = self.grid.lls2hls(st)

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

                ### ADD TRANSITIONS TO BUFFER
                for ep_idx in range(len(hl_a_list)):

                    self.memory.add(hl_s_list[ep_idx], hl_a_list[ep_idx], hl_s_list[ep_idx + 1], hl_r_list[ep_idx],
                                    hl_d_list[ep_idx], image)

                    if self.config.mvprop_her:
                        ### GET THE HINDSIGHT GOAL TRANSITION
                        image_her = np.zeros((1, 2, self.grid.x_size, self.grid.y_size))
                        image_her[0, 0, :, :] = self.grid.terrain_map_un_padded
                        image_her[0, 1, :, :] = -1 * np.ones((self.grid.x_size, self.grid.y_size))
                        image_her[0, 1, hl_s_list[-1][0], hl_s_list[-1][1]] = 0
                        image_her = torch.from_numpy(image_her).float().cuda()

                        if (hl_s_list[ep_idx + 1][0] == hl_s_list[-1][0]) and (
                                hl_s_list[ep_idx + 1][1] == hl_s_list[-1][1]):
                            hgt_reward = 0
                            hgt_done = True
                        else:
                            hgt_reward = -1
                            hgt_done = False

                        self.memory.add(hl_s_list[ep_idx], hl_a_list[ep_idx], hl_s_list[ep_idx + 1], hgt_reward,
                                        hgt_done, image_her)

                ### OPTIMIZE NETWORK PARAMETERS
                for _ in range(40):
                    self.mvprop_optimizer.train(self.config.time_horizon / self.time_scale)

                # TARGET NET UPDATE
                if self.dqn_steps % self.config.mvprop_target_update_frequency == 0:
                    tau = 0.05
                    for param, target_param in zip(self.mvprop_optimizer.mvprop.parameters(),
                                                   self.mvprop_optimizer.target_mvprop.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            return batch, total_success / j, total_wp_success / jwp, num_steps / j, num_steps / num_roll_outs

        else:
            self.grid.reset_env_terrain()
            start_pos = self.grid.sample_random_start_terrain(number=1)[0]
            goal_pos = self.grid.sample_random_goal_terrain(number=1)[0]
            s_goal = self.grid.lls2hls(goal_pos)
            s_init = self.grid.reset(start_pos, goal_pos)
            self.episode_steps = 0

            image = np.zeros((1, 2, self.grid.x_size, self.grid.y_size))
            image[0, 0, :, :] = self.grid.terrain_map_un_padded
            image[0, 1, :, :] = -1 * np.ones((self.grid.x_size, self.grid.y_size))
            image[0, 1, s_goal[0], s_goal[1]] = 0
            image = torch.from_numpy(image).float().cuda()

            with torch.no_grad():
                v = self.mvprop_optimizer.target_mvprop(image)
                v = v.cpu().detach()

            horizon_left = self.config.time_horizon
            st = start_pos
            success = False

            s_bar = self.grid.lls2hls(st)

            while (horizon_left > 0) and not success:

                # GET THE TARGET VECTOR
                with torch.no_grad():
                    options_x = [s_bar[0], s_bar[0], s_bar[0] + 1, s_bar[0] + 1, s_bar[0] + 1, s_bar[0],
                                 s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1]
                    options_y = [s_bar[1], s_bar[1] + 1, s_bar[1] + 1, s_bar[1], s_bar[1] - 1, s_bar[1] - 1,
                                 s_bar[1] - 1, s_bar[1], s_bar[1] + 1]
                    options_x = [self.bound_x(e) for e in options_x]
                    options_y = [self.bound_y(e) for e in options_y]
                    v_options = v[0, 0, options_x, options_y]
                    option = np.argmax(v_options)

                if option == 0:
                    target = (s_bar[0], s_bar[1])
                elif option == 1:
                    target = (s_bar[0], s_bar[1] + 1)
                elif option == 2:
                    target = (s_bar[0] + 1, s_bar[1] + 1)
                elif option == 3:
                    target = (s_bar[0] + 1, s_bar[1])
                elif option == 4:
                    target = (s_bar[0] + 1, s_bar[1] - 1)
                elif option == 5:
                    target = (s_bar[0], s_bar[1] - 1)
                elif option == 6:
                    target = (s_bar[0] - 1, s_bar[1] - 1)
                elif option == 7:
                    target = (s_bar[0] - 1, s_bar[1])
                elif option == 8:
                    target = (s_bar[0] - 1, s_bar[1] + 1)
                target = (max(0, min(target[0], self.grid.x_size - 1)),
                          max(0, min(target[1], self.grid.y_size - 1)))

                roll_out, states, actions, rewards, wp_success, l_state, _ = self.roll_out_in_env(
                    start=s_init,
                    goal=target,
                    horizon=self.time_scale,
                    ultimate_goal=goal_pos,
                    mode='test'
                )

                s_init = l_state

                st = l_state[:2]
                s_bar = self.grid.lls2hls(st)

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
            print(self.memory.size)
            print(self.eps)

    def test(self):
        g_success = self.simulate_env(mode='test')
        return g_success
