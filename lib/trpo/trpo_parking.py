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
import torch
import torch.nn.functional as F
import gym
import highway_env

from lib.policy.batch import Batch
from lib.policy.continuous_mlp import ContinuousMLP
from lib.optimizers.actor_critic.actor_critic import TRPO

from lib.environments.parking import ParkingEmpty


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

        self.op_mode = op_mode

        self.batch_size = batch_size
        self.inner_episodes = inner_episodes
        self.max_iter = max_iter
        self.state_dim = state_dim

    def roll_out_in_env(self, start, horizon):
        roll_out = Batch()
        s = start
        s_list = []
        a_list = []
        r_list = []

        success_var = False
        oob = False
        break_var = False

        for step_i in range(horizon):

            target_vec = s['desired_goal'] - s['observation']
            s_save = copy.deepcopy(np.array(list(target_vec[:2]) + list(s['observation'][2:4]) + list(target_vec[2:4])))

            s_list.append(copy.deepcopy(s['observation']))
            s_tensor = torch.tensor(s_save, dtype=torch.float).unsqueeze(0)
            a = self.policy.select_action(s_tensor)

            s_new, r, d, info = self.grid.env.step(a)
            success_var = info["is_success"]
            info = not info["is_success"]

            r = 0.0
            if success_var:
                r = 1.0
                info = False
                break_var = True

            break_var = break_var or (not info) or (step_i + 1 == horizon)

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

        return roll_out, s_list, a_list, r_list, success_var

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

                roll_out, _, _, _, success = self.roll_out_in_env(start=s_init, horizon=self.max_iter)

                jwp = 1.
                num_roll_outs += 1
                num_steps += roll_out.length()
                batch.append(roll_out)

                total_success += success
                j += 1

            return batch, total_success / j, total_wp_success / jwp, num_steps / j, num_steps / num_roll_outs

        else:

            """ INITIALIZE THE ENVIRONMENT """
            s_init = self.grid.reset()

            roll_out, state_list, action_list, reward_list, success = self.roll_out_in_env(start=s_init,
                                                                                           horizon=self.max_iter)

            num_roll_outs += 1
            num_steps += roll_out.length()
            batch.append(roll_out)

            total_success += success
            j += 1

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
