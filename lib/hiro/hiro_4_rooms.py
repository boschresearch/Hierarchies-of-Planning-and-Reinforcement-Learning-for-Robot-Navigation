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

import time

import lib.td3.utils as utils
from lib.td3.TD3 import TD3

import torch
import torch.nn.functional as F

from lib.policy.batch import Batch
from lib.policy.continuous_mlp import ContinuousMLP
from lib.optimizers.actor_critic.actor_critic import TRPO

from lib.environments.gridworld import GridWorldContinuous


class Manager:
    def __init__(self):
        self.s = None
        self.sg = None

    def update(self, s, sg):
        self.s = s[:2]
        self.sg = sg

    def reward(self, cs):
        return - np.linalg.norm(self.s + self.sg - cs[:2])

    def target(self, cs):
        return self.s + self.sg - cs[:2]


class HIROTrainer:
    def __init__(self,
                 iter_num,
                 iter_size,
                 state_dim,
                 action_dim,
                 max_action,
                 batch_size,
                 discount,
                 tau,
                 expl_noise,
                 policy_noise,
                 noise_clip,
                 policy_freq,
                 max_iter,
                 start_time_steps,
                 total_iter,
                 terrain_var,
                 her_var
                 ):

        self.iter_num = iter_num
        self.iter_size = iter_size

        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": discount,
            "tau": tau,
            "policy_noise": policy_noise * max_action,
            "noise_clip": noise_clip * max_action,
            "policy_freq": policy_freq
        }

        self.action_dim = action_dim

        self.grid = GridWorldContinuous(grid_mode='standard', terrain=terrain_var)
        self.grid.reset_env_terrain()

        self.manager_policy = TD3(**kwargs)
        self.replay_buffer = utils.ReplayBufferHIRO(state_dim, action_dim, max_size=int(2e5))

        self.policy = ContinuousMLP(state_dim,
                                    action_dim,
                                    hidden_sizes=[64, 64, 64],
                                    activation=F.relu,
                                    is_disc_action=False)
        self.optimizer = TRPO(policy=self.policy,
                              use_gpu=False,
                              max_kl=5e-4,
                              damping=5e-3,
                              use_fim=False,
                              discount=0.99,
                              imp_weight=False)

        self.manager = Manager()

        self.batch_size = batch_size
        self.expl_noise = expl_noise
        self.expl_noise_start = expl_noise
        self.max_action = max_action

        self.total_steps = 0
        self.max_iter = max_iter
        self.start_time_steps = start_time_steps

        self.manager_time_scale = 2

        self.current_iter = 0
        self.total_iter = total_iter

        self.her_var = her_var

    def roll_out_in_env(self, start, goal, horizon, mode='train'):
        roll_out = Batch()
        done = False
        s = self.grid.reset(start, goal)
        s_list = []
        a_list = []
        r_list = []

        state_seq = []
        action_seq = []

        s_bar = self.grid.lls2hls(s)
        s_manager = copy.deepcopy(
            np.array(list(s) + list(self.grid.state_cache[s_bar[0], s_bar[1]]) + list(goal)))
        r_manager = 0.
        if mode == 'train':
            if self.total_steps < self.start_time_steps:
                sub_goal = (4 * np.random.random((2,))) - 2
            else:
                sub_goal = (self.manager_policy.select_action(s_manager) +
                            np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)).\
                    clip(-self.max_action, self.max_action)
        else:
            sub_goal = self.manager_policy.select_action(s_manager)

        self.manager.update(s=copy.deepcopy(s), sg=sub_goal)

        s_goal = self.grid.lls2hls(goal)

        for step_i in range(horizon):

            if mode == 'train':
                self.total_steps += 1

            s_bar = self.grid.lls2hls(s)
            s_save = copy.deepcopy(np.array(list(s) + list(self.grid.state_cache[s_bar[0], s_bar[1]]) +
                                            list(self.manager.target(s))))
            s_list.append(s_save)

            s_tensor = torch.tensor(s_save, dtype=torch.float).unsqueeze(0)
            a = self.policy.select_action(s_tensor)

            state_seq.append(s_save)
            action_seq.append(a)

            s_new, _, _, _, _ = self.grid.step(action=10*a)
            r = self.manager.reward(s_new)
            a_list.append(a)
            r_list.append(r)

            s_new_bar = self.grid.lls2hls(s_new)
            done = ((s_new_bar[0] == s_goal[0]) and (s_new_bar[1] == s_goal[1]))
            r_manager += -1 * float(not done)

            manager_update = (step_i + 1) % self.manager_time_scale == 0

            roll_out.append(
                Batch([a.astype(np.float32)],
                      [s_save.astype(np.float32)],
                      [r],
                      [s_new.astype(np.float32)],
                      [0 if ((step_i + 1 == horizon) or done or manager_update) else 1],
                      [not done],
                      [1.0]))

            s = s_new

            if manager_update or done:
                self.total_steps += 1
                s_new_manager = copy.deepcopy(np.array(list(s) +
                                                       list(self.grid.state_cache[s_new_bar[0], s_new_bar[1]]) +
                                                       list(goal)))
                if mode == 'train':
                    self.replay_buffer.add(s_manager, self.manager.sg, s_new_manager, r_manager, done,
                                           np.array(state_seq), np.array(action_seq))
                    if self.her_var:
                        s_manager_her = np.concatenate((s_manager[:12], s[:2]))
                        s_new_manager_her = np.concatenate((s_new_manager[:12], s[:2]))
                        r_manager_her = -len(state_seq) + 1
                        self.replay_buffer.add(s_manager_her, self.manager.sg, s_new_manager_her, r_manager_her, True,
                                               np.array(state_seq), np.array(action_seq))

                    state_seq = []
                    action_seq = []

                s_manager = s_new_manager
                r_manager = 0.

                if mode == 'train':
                    if self.total_steps < self.start_time_steps:
                        sub_goal = (4 * np.random.random((2,))) - 2
                    else:
                        sub_goal = (self.manager_policy.select_action(s_manager) +
                                    np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)). \
                            clip(-self.max_action, self.max_action)
                else:
                    sub_goal = self.manager_policy.select_action(s_manager)

                self.manager.update(s=copy.deepcopy(s), sg=sub_goal)

            if done:
                break
        s_save = copy.deepcopy(np.array(list(s) + list(self.grid.state_cache[s_bar[0], s_bar[1]]) +
                                        list(self.manager.target(s))))
        s_list.append(s_save)
        return roll_out, step_i + 1, s_list, a_list, r_list, done

    def simulate_env(self, mode):
        batch = Batch()
        num_roll_outs = 0
        num_steps = 0
        total_success = 0

        if mode == 'train':
            while num_steps < self.iter_size:
                self.grid.reset_env_terrain()
                start_pos = self.grid.sample_random_start_terrain(number=1)[0]
                goal_pos = self.grid.sample_random_goal_terrain(number=1)[0]

                roll_out, steps, states, actions, rewards, success = self.roll_out_in_env(start=start_pos,
                                                                                          goal=goal_pos,
                                                                                          horizon=self.max_iter,
                                                                                          mode='train')

                if self.total_steps > self.start_time_steps:
                    for _ in range(40):
                        self.manager_policy.train(self.replay_buffer, self.batch_size)

                batch.append(roll_out)
                num_roll_outs += 1
                num_steps += steps

                total_success += success

            return batch, total_success / num_roll_outs, num_steps / num_roll_outs

        else:
            self.grid.reset_env_terrain()
            start_pos = self.grid.sample_random_start_terrain(number=1)[0]
            goal_pos = self.grid.sample_random_goal_terrain(number=1)[0]

            _, steps, states, actions, rewards, success = self.roll_out_in_env(start=start_pos,
                                                                               goal=goal_pos,
                                                                               horizon=self.max_iter,
                                                                               mode='test')

        return success

    def train(self):
        for iter_loop in range(self.iter_num):
            self.current_iter += 1
            self.expl_noise = max((1 - (self.current_iter / self.total_iter)) * self.expl_noise_start, 0)
            batch, g_success, steps_trajectory = self.simulate_env(mode='train')
            self.optimizer.process_batch(self.policy, batch, [])
            print("OC")
            print(self.replay_buffer.size)
            start_t = time.time()
            self.replay_buffer.off_policy_correction(policy=self.policy)
            print(time.time() - start_t)
            print("STATS")
            print(g_success)
            print(steps_trajectory)

    def test(self):
        success = self.simulate_env(mode='test')
        return success
