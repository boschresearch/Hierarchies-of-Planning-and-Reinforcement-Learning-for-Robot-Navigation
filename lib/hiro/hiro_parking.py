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

import time

import gym
import highway_env

import lib.td3.utils as utils
from lib.td3.TD3_difflim import TD3

import torch
import torch.nn.functional as F

from lib.policy.batch import Batch
from lib.policy.continuous_mlp import ContinuousMLP
from lib.optimizers.actor_critic.actor_critic import TRPO

from lib.environments.parking import ParkingEmpty


class ManagerAngle:
    def __init__(self):
        self.s = None
        self.a = None
        self.sg = None

    def update(self, s, sg):
        self.s = np.array([s[0], s[1], np.arctan2(s[5], s[4])])
        self.a = sg

    def reward(self, cs):
        cs_ang = np.array([cs[0], cs[1], np.arctan2(cs[5], cs[4])])
        diff_ang = self.s + self.a - cs_ang
        if diff_ang[2] > math.pi:
            while diff_ang[2] > math.pi:
                diff_ang[2] -= 2 * math.pi
        if diff_ang[2] < -math.pi:
            while diff_ang[2] < -math.pi:
                diff_ang[2] += 2 * math.pi
        return - np.linalg.norm(np.dot(diff_ang, np.array([25, 25, 1 / math.pi])))

    def target(self, cs):
        cs_ang = np.array([cs[0], cs[1], np.arctan2(cs[5], cs[4])])
        diff_ang = self.s + self.a - cs_ang
        return np.array([diff_ang[0], diff_ang[1], np.cos(diff_ang[2]), np.sin(diff_ang[2])])

    def target_ang(self, cs):
        cs_ang = np.array([cs[0], cs[1], np.arctan2(cs[5], cs[4])])
        diff_ang = self.s + self.a - cs_ang
        return [diff_ang[2]]


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
                 total_iter
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

        self.grid = ParkingEmpty()

        self.manager_policy = TD3(**kwargs)
        self.replay_buffer = utils.ReplayBufferHIROParkingAngleNew(state_dim, action_dim, max_size=int(2e5))

        self.policy = ContinuousMLP(6,
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

        self.manager = ManagerAngle()

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

    def roll_out_in_env(self, horizon, mode='train'):
        roll_out = Batch()
        oob = False
        s = self.grid.reset()
        s_list = []
        a_list = []
        r_list = []

        state_seq = []
        action_seq = []

        ultimate_start_bar = self.grid.lls2hls(s['observation'])
        goal_bar = self.grid.lls2hls(s['desired_goal'])

        s_manager = copy.deepcopy(np.concatenate((s['observation'], s['desired_goal'])))
        r_manager = 0.
        if mode == 'train':
            if self.total_steps < self.start_time_steps:
                sub_goal = np.concatenate((0.08 * np.random.random((2, )) - 0.04, 2 * math.pi * np.random.random((1, )) - math.pi))
            else:
                sub_goal = (self.manager_policy.select_action(s_manager) +
                            np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)).\
                    clip(-self.max_action, self.max_action)
        else:
            sub_goal = self.manager_policy.select_action(s_manager)

        self.manager.update(s=copy.deepcopy(s['observation']), sg=sub_goal)

        for step_i in range(horizon):

            if mode == 'train':
                self.total_steps += 1

            manager_target = self.manager.target(s['observation'])
            s_save = copy.deepcopy(
                np.array(list(manager_target[:2]) + list(s['observation'][2:4]) + list(manager_target[2:4])))
            s_list.append(s_save)

            s_tensor = torch.tensor(s_save, dtype=torch.float).unsqueeze(0)
            a = self.policy.select_action(s_tensor)

            state_seq.append(copy.deepcopy(np.array(list(manager_target[:2]) + list(s['observation'][2:4]) +
                                                    self.manager.target_ang(s['observation']))))

            action_seq.append(a)

            s_new, r, d, info = self.grid.env.step(a)
            info = info["is_success"]
            r = self.manager.reward(s_new['observation'])
            a_list.append(a)
            r_list.append(r)
            r_manager += float(info)

            manager_update = (step_i + 1) % self.manager_time_scale == 0

            ib = self.grid.check_in_bounds(s_new['observation'])
            if not ib:
                oob = True

            roll_out.append(
                Batch([a.astype(np.float32)],
                      [s_save.astype(np.float32)],
                      [r],
                      [s_new['observation'].astype(np.float32)],
                      [0 if ((step_i + 1 == horizon) or info or oob or manager_update) else 1],
                      [not info],
                      [1.0]))

            s = s_new

            if manager_update or info:
                self.total_steps += 1
                s_new_manager = copy.deepcopy(np.concatenate((s['observation'], s['desired_goal'])))

                if mode == 'train':
                    self.replay_buffer.add(s_manager, self.manager.a, s_new_manager, r_manager, info,
                                           np.array(state_seq), np.array(action_seq))

                    s_manager_her = np.concatenate((s_manager[:6], s['observation']))
                    s_new_manager_her = np.concatenate((s_new_manager[:6], s['observation']))
                    self.replay_buffer.add(s_manager_her, self.manager.a, s_new_manager_her, 1.0, True,
                                           np.array(state_seq), np.array(action_seq))

                    state_seq = []
                    action_seq = []

                s_manager = s_new_manager
                r_manager = 0.

                if mode == 'train':
                    if self.total_steps < self.start_time_steps:
                        sub_goal = np.concatenate(
                            (0.08 * np.random.random((2,)) - 0.04, 2 * math.pi * np.random.random((1,)) - math.pi))
                    else:
                        sub_goal = (self.manager_policy.select_action(s_manager) +
                                    np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)). \
                            clip(-self.max_action, self.max_action)
                else:
                    sub_goal = self.manager_policy.select_action(s_manager)

                self.manager.update(s=copy.deepcopy(s['observation']), sg=sub_goal)

            if info or oob:
                break
        manager_target = self.manager.target(s['observation'])
        s_save = copy.deepcopy(np.array(list(manager_target[:2]) + list(s['observation'][2:4]) +
                                        list(manager_target[2:4])))
        s_list.append(s_save)
        return roll_out, step_i + 1, s_list, a_list, r_list, info, ultimate_start_bar, goal_bar

    def simulate_env(self, mode):
        batch = Batch()
        num_roll_outs = 0
        num_steps = 0
        total_success = 0

        if mode == 'train':
            while num_steps < self.iter_size:

                roll_out, steps, states, actions, rewards, success, start_pos, goal_pos = self.roll_out_in_env(
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
            _, steps, states, actions, rewards, success, start_pos, goal_pos = self.roll_out_in_env(
                horizon=self.max_iter,
                mode='test')

        return success

    def train(self):
        for iter_loop in range(self.iter_num):
            self.current_iter += 1
            self.expl_noise = max((1 - (self.current_iter / (self.total_iter))) * self.expl_noise_start, 0)
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
