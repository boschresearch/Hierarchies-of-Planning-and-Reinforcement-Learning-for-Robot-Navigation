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

import torch
import torch.nn.functional as F
import numpy as np

from lib.policy.batch import Batch
from lib.policy.continuous_mlp import ContinuousMLP
from lib.optimizers.actor_critic.actor_critic import TRPO

from lib.environments.gridworld import GridWorldContinuous


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
                                        waypoint=True,
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

        self.env_mode = env_mode
        self.op_mode = op_mode
        self.state_dim = state_dim
        self.episode_steps = 0

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
            target_vec = self.grid.goal_management.get_target_vec(s[:2])
            s_save = copy.deepcopy(np.array(list(s[2:]) + list(self.grid.state_cache[s_bar[0], s_bar[1]]) +
                                            list(target_vec)))
            s_pos_save = copy.deepcopy(np.array(s[:2]))
            s_list.append(np.concatenate((s_pos_save, s_save)))
            s = torch.tensor(s_save, dtype=torch.float).unsqueeze(0)
            a = self.policy.select_action(s)

            s_new, r, d, _, info = self.grid.step(action=10*a)
            s_bar_cand = self.grid.lls2hls(s_new)

            success_var = ((s_bar_cand[0] == s_u_goal[0]) and (s_bar_cand[1] == s_u_goal[1]))
            if success_var:
                info = False
                break_var = True

            break_var_rrt = self.grid.goal_management.update_way_point(self.grid, (s_new[0], s_new[1]), d)
            break_var = break_var or break_var_rrt or (not info) or (step_i + 1 == horizon)

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

                path_segment_len_list = self.grid.goal_management.reset(start_pos, goal_pos, self.grid)
                self.grid.old_distance = self.grid.goal_management.path_segment_len_list[0]

                """ START THE EPISODE """
                horizon_left = self.config.time_horizon
                success = False

                while (horizon_left > 0) and not success:

                    # GET THE TARGET VECTOR
                    curr_goal = \
                        self.grid.goal_management.way_points[self.grid.goal_management.way_point_current]

                    roll_out, _, _, _, wp_success, l_state, s_bar_p = self.roll_out_in_env(
                        start=s_init,
                        goal=curr_goal,
                        horizon=horizon_left,
                        ultimate_goal=goal_pos,
                        mode='train'
                    )

                    s_init = l_state

                    num_roll_outs += 1
                    num_steps += roll_out.length()
                    horizon_left -= roll_out.length()

                    total_wp_success += wp_success
                    jwp += 1

                    st_bar = self.grid.lls2hls(l_state)
                    success = ((st_bar[0] == s_goal[0]) and (st_bar[1] == s_goal[1]))

                    batch.append(roll_out)

                total_success += success
                j += 1

            return batch, total_success / j, total_wp_success / jwp, num_steps / j, num_steps / num_roll_outs

        else:
            self.grid.reset_env_terrain()
            start_pos = self.grid.sample_random_start_terrain(number=1)[0]
            goal_pos = self.grid.sample_random_goal_terrain(number=1)[0]
            s_goal = self.grid.lls2hls(goal_pos)
            s_init = self.grid.reset(start_pos, goal_pos)
            self.episode_steps = 0

            path_segment_len_list = self.grid.goal_management.reset(start_pos, goal_pos, self.grid)
            self.grid.old_distance = self.grid.goal_management.path_segment_len_list[0]

            """ START THE EPISODE """
            horizon_left = self.config.time_horizon
            success = False

            while (horizon_left > 0) and not success:
                # GET THE TARGET VECTOR
                curr_goal = \
                    self.grid.goal_management.way_points[self.grid.goal_management.way_point_current]

                roll_out, states, actions, rewards, wp_success, l_state, _ = self.roll_out_in_env(
                    start=s_init,
                    goal=curr_goal,
                    horizon=horizon_left,
                    ultimate_goal=goal_pos,
                    mode='test'
                )

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

    def test(self):
        g_success = self.simulate_env(mode='test')
        return g_success
