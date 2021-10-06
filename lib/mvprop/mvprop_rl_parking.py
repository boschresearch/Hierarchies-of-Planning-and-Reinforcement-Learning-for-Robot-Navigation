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

from lib.mvprop.mvprop import MVPROPFAT3D
from lib.mvprop.mvprop_optimizer import MVPROPOptimizer3D
from lib.td3.utils import ReplayBufferVIN3D


def target_2_target_vec(target, state):
    if target[2] == 0:
        angle = 0
    elif target[2] == 1:
        angle = np.pi / 4
    elif target[2] == 2:
        angle = np.pi / 2
    elif target[2] == 3:
        angle = 3 * np.pi / 4
    elif target[2] == 4:
        angle = np.pi
    elif target[2] == 5:
        angle = -3 * np.pi / 4
    elif target[2] == 6:
        angle = -np.pi / 2
    elif target[2] == 7:
        angle = -np.pi / 4
    else:
        print(target)
    ch = np.cos(angle)
    sh = np.sin(angle)
    return (target[0] / 25) - 0.46 - state[0], (target[1] / 25) - 0.22 - state[1], ch - state[4], sh - state[5]


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
        self.mvprop = MVPROPFAT3D(k=100).cuda()
        self.mvprop_target = copy.deepcopy(self.mvprop).cuda()
        self.memory = ReplayBufferVIN3D(3, 1, self.grid.x_size, 35000)
        self.mvprop_optimizer = MVPROPOptimizer3D(self.mvprop,
                                                  self.mvprop_target,
                                                  self.memory,
                                                  0.99,
                                                  128,
                                                  3e-4,
                                                  100)

        self.op_mode = op_mode

        self.batch_size = batch_size
        self.inner_episodes = inner_episodes
        self.max_iter = max_iter
        self.state_dim = state_dim
        self.dqn_steps = 0
        self.eps = 1.0
        self.time_scale = 2
        self.episode_steps = 0

    def bound_x(self, input_var):
        return max(0, min(input_var, self.grid.x_size - 1))

    def bound_y(self, input_var):
        return max(0, min(input_var, self.grid.y_size - 1))

    def roll_out_in_env(self, start, target, horizon):
        roll_out = Batch()
        s = start
        s_list = []
        a_list = []
        r_list = []

        d = False
        oob = False
        break_var = False
        target_vec = target_2_target_vec(target, s['observation'])
        goal = self.grid.lls2hls(s['observation'] + np.array(list(target_vec[:2]) + [0, 0] + list(target_vec[2:4])))

        for step_i in range(horizon):
            self.episode_steps += 1

            target_vec = target_2_target_vec(target, s['observation'])
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

                """ IMAGE INPUT """
                image = np.zeros((1, 2, self.grid.x_size, self.grid.y_size, 8))
                image[0, 0, :, :, :] = np.ones((self.grid.x_size, self.grid.y_size, 8))
                image[0, 1, :, :, :] = -1 * np.ones((self.grid.x_size, self.grid.y_size, 8))
                image[0, 1, s_goal[0], s_goal[1], s_goal[2]] = 0
                image = torch.from_numpy(image).float().cuda()

                with torch.no_grad():
                    v = self.mvprop_optimizer.mvprop(image)
                    v = v.cpu().detach()

                """ START THE EPISODE """
                horizon_left = self.max_iter
                success = False

                s_bar = self.grid.lls2hls(s_init['observation'])
                hl_s_list = []
                hl_a_list = []
                hl_r_list = []
                hl_d_list = []
                hl_s_list.append(s_bar)

                while (horizon_left > 0) and not success:

                    # GET THE TARGET VECTOR
                    self.dqn_steps += 1
                    self.eps = 0.01 + 0.99 * math.exp(-1. * self.dqn_steps / 10000)
                    s_bar = self.grid.lls2hls(s_init['observation'])
                    hl_s_list.append(s_bar)

                    if torch.rand(1)[0] > self.eps:
                        with torch.no_grad():
                            options_x = [s_bar[0], s_bar[0], s_bar[0] + 1, s_bar[0] + 1, s_bar[0] + 1, s_bar[0],
                                         s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1, s_bar[0], s_bar[0], s_bar[0] + 1,
                                         s_bar[0] + 1, s_bar[0] + 1, s_bar[0], s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1,
                                         s_bar[0], s_bar[0], s_bar[0] + 1, s_bar[0] + 1, s_bar[0] + 1, s_bar[0],
                                         s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1, s_bar[0], s_bar[0], s_bar[0] + 1,
                                         s_bar[0] + 1, s_bar[0] + 1, s_bar[0], s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1,
                                         s_bar[0], s_bar[0], s_bar[0] + 1, s_bar[0] + 1, s_bar[0] + 1, s_bar[0],
                                         s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1, s_bar[0], s_bar[0], s_bar[0] + 1,
                                         s_bar[0] + 1, s_bar[0] + 1, s_bar[0], s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1,
                                         s_bar[0], s_bar[0], s_bar[0] + 1, s_bar[0] + 1, s_bar[0] + 1, s_bar[0],
                                         s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1, s_bar[0], s_bar[0], s_bar[0] + 1,
                                         s_bar[0] + 1, s_bar[0] + 1, s_bar[0], s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1]
                            options_y = [s_bar[1], s_bar[1] + 1, s_bar[1] + 1, s_bar[1], s_bar[1] - 1, s_bar[1] - 1,
                                         s_bar[1] - 1, s_bar[1], s_bar[1] + 1, s_bar[1], s_bar[1] + 1, s_bar[1] + 1,
                                         s_bar[1], s_bar[1] - 1, s_bar[1] - 1, s_bar[1] - 1, s_bar[1], s_bar[1] + 1,
                                         s_bar[1], s_bar[1] + 1, s_bar[1] + 1, s_bar[1], s_bar[1] - 1, s_bar[1] - 1,
                                         s_bar[1] - 1, s_bar[1], s_bar[1] + 1, s_bar[1], s_bar[1] + 1, s_bar[1] + 1,
                                         s_bar[1], s_bar[1] - 1, s_bar[1] - 1, s_bar[1] - 1, s_bar[1], s_bar[1] + 1,
                                         s_bar[1], s_bar[1] + 1, s_bar[1] + 1, s_bar[1], s_bar[1] - 1, s_bar[1] - 1,
                                         s_bar[1] - 1, s_bar[1], s_bar[1] + 1, s_bar[1], s_bar[1] + 1, s_bar[1] + 1,
                                         s_bar[1], s_bar[1] - 1, s_bar[1] - 1, s_bar[1] - 1, s_bar[1], s_bar[1] + 1,
                                         s_bar[1], s_bar[1] + 1, s_bar[1] + 1, s_bar[1], s_bar[1] - 1, s_bar[1] - 1,
                                         s_bar[1] - 1, s_bar[1], s_bar[1] + 1, s_bar[1], s_bar[1] + 1, s_bar[1] + 1,
                                         s_bar[1], s_bar[1] - 1, s_bar[1] - 1, s_bar[1] - 1, s_bar[1], s_bar[1] + 1]
                            options_z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                                         2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5,
                                         5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7]
                            options_x = [self.bound_x(e) for e in options_x]
                            options_y = [self.bound_y(e) for e in options_y]
                            v_options = v[0, 0, options_x, options_y, options_z]
                            option = np.argmax(v_options)
                    else:
                        option = randint(0, 71)

                    option_o = np.floor(option / 9)
                    option_p = option % 9

                    if option_p == 0:
                        target_p = (s_bar[0], s_bar[1])
                    elif option_p == 1:
                        target_p = (s_bar[0], s_bar[1] + 1)
                    elif option_p == 2:
                        target_p = (s_bar[0] + 1, s_bar[1] + 1)
                    elif option_p == 3:
                        target_p = (s_bar[0] + 1, s_bar[1])
                    elif option_p == 4:
                        target_p = (s_bar[0] + 1, s_bar[1] - 1)
                    elif option_p == 5:
                        target_p = (s_bar[0], s_bar[1] - 1)
                    elif option_p == 6:
                        target_p = (s_bar[0] - 1, s_bar[1] - 1)
                    elif option_p == 7:
                        target_p = (s_bar[0] - 1, s_bar[1])
                    elif option_p == 8:
                        target_p = (s_bar[0] - 1, s_bar[1] + 1)
                    target_p = (max(0, min(target_p[0], self.grid.x_size - 1)),
                                max(0, min(target_p[1], self.grid.y_size - 1)))
                    target = (target_p[0], target_p[1], int(option_o))

                    roll_out, _, _, _, wp_success, success, l_state, s_bar_p, oob = self.roll_out_in_env(
                        horizon=self.time_scale,
                        start=s_init,
                        target=target)

                    s_init = l_state

                    hl_s_list.append(s_bar_p)
                    hl_a_list.append(option)

                    num_roll_outs += 1
                    num_steps += roll_out.length()
                    horizon_left -= roll_out.length()

                    total_wp_success += wp_success
                    jwp += 1

                    st_bar = self.grid.lls2hls(l_state['observation'])
                    success_tile = ((st_bar[0] == s_goal[0]) and (st_bar[1] == s_goal[1]) and (st_bar[2] == s_goal[2]))
                    if success_tile:
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

                ### ADD TRANSITIONS TO BUFFER
                for ep_idx in range(len(hl_a_list)):

                    self.memory.add(hl_s_list[ep_idx], hl_a_list[ep_idx], hl_s_list[ep_idx + 1], hl_r_list[ep_idx],
                                    hl_d_list[ep_idx], image)

                    if True:
                        ##### GET THE HINDSIGHT GOAL TRANSITION
                        image_her = np.zeros((1, 2, self.grid.x_size, self.grid.y_size, 8))
                        image_her[0, 0, :, :, :] = np.ones((self.grid.x_size, self.grid.y_size, 8))
                        image_her[0, 1, :, :, :] = -1 * np.ones((self.grid.x_size, self.grid.y_size, 8))
                        image_her[0, 1, hl_s_list[-1][0], hl_s_list[-1][1], hl_s_list[-1][2]] = 0
                        image_her = torch.from_numpy(image_her).float().cuda()

                        if (hl_s_list[ep_idx + 1][0] == hl_s_list[-1][0]) and \
                                (hl_s_list[ep_idx + 1][1] == hl_s_list[-1][1]) and \
                                (hl_s_list[ep_idx + 1][2] == hl_s_list[-1][2]):
                            hgt_reward = 0
                            hgt_done = True
                        else:
                            hgt_reward = -1
                            hgt_done = False

                        self.memory.add(hl_s_list[ep_idx], hl_a_list[ep_idx], hl_s_list[ep_idx + 1], hgt_reward,
                                        hgt_done, image_her)

                ### OPTIMIZE NETWORK PARAMETERS
                for _ in range(40):
                    self.mvprop_optimizer.train(self.max_iter / self.time_scale)

                # TARGET NET UPDATE
                if self.dqn_steps % 1 == 0:
                    tau = 0.05
                    for param, target_param in zip(self.mvprop_optimizer.mvprop.parameters(),
                                                   self.mvprop_optimizer.target_mvprop.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            return batch, total_success / j, total_wp_success / jwp, num_steps / j, num_steps / num_roll_outs

        else:

            """ INITIALIZE THE ENVIRONMENT """
            s_init = self.grid.reset()
            s_start = self.grid.lls2hls(s_init['observation'])
            s_goal = self.grid.lls2hls(s_init['desired_goal'])
            self.episode_steps = 0

            """ IMAGE INPUT """
            image = np.zeros((1, 2, self.grid.x_size, self.grid.y_size, 8))
            image[0, 0, :, :, :] = np.ones((self.grid.x_size, self.grid.y_size, 8))
            image[0, 1, :, :, :] = -1 * np.ones((self.grid.x_size, self.grid.y_size, 8))
            image[0, 1, s_goal[0], s_goal[1], s_goal[2]] = 0
            image = torch.from_numpy(image).float().cuda()

            with torch.no_grad():
                v = self.mvprop_optimizer.target_mvprop(image)
                v = v.cpu().detach()

            """ START THE EPISODE """
            horizon_left = self.max_iter
            success = False

            s_bar = self.grid.lls2hls(s_init['observation'])
            hl_s_list = []
            hl_a_list = []
            hl_r_list = []
            hl_d_list = []
            hl_s_list.append(s_bar)

            while (horizon_left > 0) and not success:

                # GET THE TARGET VECTOR
                s_bar = self.grid.lls2hls(s_init['observation'])
                hl_s_list.append(s_bar)

                with torch.no_grad():
                    options_x = [s_bar[0], s_bar[0], s_bar[0] + 1, s_bar[0] + 1, s_bar[0] + 1, s_bar[0],
                                 s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1, s_bar[0], s_bar[0], s_bar[0] + 1,
                                 s_bar[0] + 1, s_bar[0] + 1, s_bar[0], s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1,
                                 s_bar[0], s_bar[0], s_bar[0] + 1, s_bar[0] + 1, s_bar[0] + 1, s_bar[0],
                                 s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1, s_bar[0], s_bar[0], s_bar[0] + 1,
                                 s_bar[0] + 1, s_bar[0] + 1, s_bar[0], s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1,
                                 s_bar[0], s_bar[0], s_bar[0] + 1, s_bar[0] + 1, s_bar[0] + 1, s_bar[0],
                                 s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1, s_bar[0], s_bar[0], s_bar[0] + 1,
                                 s_bar[0] + 1, s_bar[0] + 1, s_bar[0], s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1,
                                 s_bar[0], s_bar[0], s_bar[0] + 1, s_bar[0] + 1, s_bar[0] + 1, s_bar[0],
                                 s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1, s_bar[0], s_bar[0], s_bar[0] + 1,
                                 s_bar[0] + 1, s_bar[0] + 1, s_bar[0], s_bar[0] - 1, s_bar[0] - 1, s_bar[0] - 1]
                    options_y = [s_bar[1], s_bar[1] + 1, s_bar[1] + 1, s_bar[1], s_bar[1] - 1, s_bar[1] - 1,
                                 s_bar[1] - 1, s_bar[1], s_bar[1] + 1, s_bar[1], s_bar[1] + 1, s_bar[1] + 1,
                                 s_bar[1], s_bar[1] - 1, s_bar[1] - 1, s_bar[1] - 1, s_bar[1], s_bar[1] + 1,
                                 s_bar[1], s_bar[1] + 1, s_bar[1] + 1, s_bar[1], s_bar[1] - 1, s_bar[1] - 1,
                                 s_bar[1] - 1, s_bar[1], s_bar[1] + 1, s_bar[1], s_bar[1] + 1, s_bar[1] + 1,
                                 s_bar[1], s_bar[1] - 1, s_bar[1] - 1, s_bar[1] - 1, s_bar[1], s_bar[1] + 1,
                                 s_bar[1], s_bar[1] + 1, s_bar[1] + 1, s_bar[1], s_bar[1] - 1, s_bar[1] - 1,
                                 s_bar[1] - 1, s_bar[1], s_bar[1] + 1, s_bar[1], s_bar[1] + 1, s_bar[1] + 1,
                                 s_bar[1], s_bar[1] - 1, s_bar[1] - 1, s_bar[1] - 1, s_bar[1], s_bar[1] + 1,
                                 s_bar[1], s_bar[1] + 1, s_bar[1] + 1, s_bar[1], s_bar[1] - 1, s_bar[1] - 1,
                                 s_bar[1] - 1, s_bar[1], s_bar[1] + 1, s_bar[1], s_bar[1] + 1, s_bar[1] + 1,
                                 s_bar[1], s_bar[1] - 1, s_bar[1] - 1, s_bar[1] - 1, s_bar[1], s_bar[1] + 1]
                    options_z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
                                 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5,
                                 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7]
                    options_x = [self.bound_x(e) for e in options_x]
                    options_y = [self.bound_y(e) for e in options_y]
                    v_options = v[0, 0, options_x, options_y, options_z]
                    option = np.argmax(v_options)

                option_o = np.floor(option / 9)
                option_p = option % 9

                if option_p == 0:
                    target_p = (s_bar[0], s_bar[1])
                elif option_p == 1:
                    target_p = (s_bar[0], s_bar[1] + 1)
                elif option_p == 2:
                    target_p = (s_bar[0] + 1, s_bar[1] + 1)
                elif option_p == 3:
                    target_p = (s_bar[0] + 1, s_bar[1])
                elif option_p == 4:
                    target_p = (s_bar[0] + 1, s_bar[1] - 1)
                elif option_p == 5:
                    target_p = (s_bar[0], s_bar[1] - 1)
                elif option_p == 6:
                    target_p = (s_bar[0] - 1, s_bar[1] - 1)
                elif option_p == 7:
                    target_p = (s_bar[0] - 1, s_bar[1])
                elif option_p == 8:
                    target_p = (s_bar[0] - 1, s_bar[1] + 1)
                target_p = (max(0, min(target_p[0], self.grid.x_size - 1)),
                            max(0, min(target_p[1], self.grid.y_size - 1)))
                target = (target_p[0], target_p[1], int(option_o))

                roll_out, _, _, _, wp_success, success, l_state, s_bar_p, oob = self.roll_out_in_env(
                    horizon=self.time_scale,
                    start=s_init,
                    target=target)

                s_init = l_state

                hl_s_list.append(s_bar_p)
                hl_a_list.append(option)

                num_roll_outs += 1
                num_steps += roll_out.length()
                horizon_left -= roll_out.length()

                total_wp_success += wp_success
                jwp += 1

                st_bar = self.grid.lls2hls(l_state['observation'])
                success_tile = ((st_bar[0] == s_goal[0]) and (st_bar[1] == s_goal[1]) and (st_bar[2] == s_goal[2]))
                if success_tile:
                    hl_r_list.append(0)
                    hl_d_list.append(True)
                else:
                    hl_r_list.append(-1)
                    hl_d_list.append(False)

                if oob:
                    break

            j = 1.

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