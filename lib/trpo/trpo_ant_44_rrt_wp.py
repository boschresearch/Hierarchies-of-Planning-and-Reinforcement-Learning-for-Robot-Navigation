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

from lib.policy.batch import Batch
from lib.policy.continuous_mlp import ContinuousMLP
from lib.optimizers.actor_critic.actor_critic import TRPO

from lib.environments.mujoco_grid import Ant44Env0

import xml.etree.ElementTree as ET


class TRPOTrainer:

    def __init__(self,
                 env_mode,
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
                 use_gpu=False,
                 mj_ant_path=None,

                 ):

        self.mj_ant_path = mj_ant_path
        self.grid = Ant44Env0(mj_ant_path=self.mj_ant_path,
                              waypoint=True)

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

        self.env_mode = env_mode
        self.op_mode = op_mode

        self.batch_size = batch_size
        self.inner_episodes = inner_episodes
        self.max_iter = max_iter
        self.state_dim = state_dim

    def roll_out_in_env(self, start, goal, ultimate_goal, horizon):
        roll_out = Batch()
        s_u_goal = self.grid.lls2hls(ultimate_goal)
        s = start

        s_list = []
        a_list = []
        r_list = []
        d = False

        for step_i in range(horizon):
            s_bar = self.grid.lls2hls(s)
            target_vec = self.grid.goal_management.get_target_vec(s[:2])

            s_save = copy.deepcopy(np.array(list(s[2:]) + list(self.grid.state_cache[s_bar[0], s_bar[1]]) +
                                            list(target_vec)))
            s_pos_save = copy.deepcopy(np.array(s[:2]))
            s_list.append(np.concatenate((s_pos_save, s_save)))
            s = torch.tensor(s_save, dtype=torch.float).unsqueeze(0)
            a = self.policy.select_action(s)

            s_new, r, d, info = self.grid.step(a, goal)
            info = info['no_goal_reached']

            dist_wp = np.sqrt(
                (s_new[0] - self.grid.goal_management.way_points[self.grid.goal_management.way_point_current][
                    0]) ** 2 +
                (s_new[1] - self.grid.goal_management.way_points[self.grid.goal_management.way_point_current][
                    1]) ** 2)
            d_wp = dist_wp < 0.5
            new_distance = dist_wp
            d = np.sqrt((s_new[0] - goal[0]) ** 2 + (s_new[1] - goal[1]) ** 2) < 0.5
            if d:
                info = False
            else:
                info = True
            r = d_wp  # + self.grid.old_distance - 0.99 * new_distance
            self.grid.old_distance = new_distance

            s_bar_cand = self.grid.lls2hls(s_new)

            break_var = self.grid.goal_management.update_way_point(self.grid, (s_new[0], s_new[1]), d)
            success_var = ((s_bar_cand[0] == s_u_goal[0]) and (s_bar_cand[1] == s_u_goal[1]))
            if success_var:
                info = False
                break_var = True

            a_list.append(a)
            r_list.append(r)
            roll_out.append(
                Batch([a.astype(np.float32)],
                      [s_save.astype(np.float32)],
                      [r],
                      [s_new.astype(np.float32)],
                      [0 if ((not info) or (step_i + 1 == horizon) or break_var) else 1],
                      [info],
                      [1.0]))
            s = s_new
            if (not info) or break_var:
                break
        return roll_out, s_list, a_list, r_list, d, s_new

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
                a, b, c, d, e, f, g, h, p = self.grid.reset_env_random()
                structure = copy.deepcopy(a)
                objects = copy.deepcopy(b)
                obstacle_list = copy.deepcopy(c)
                occupancy_map = copy.deepcopy(d)
                default_starts = copy.deepcopy(e)
                state_cache = copy.deepcopy(f)
                occupancy_map_padded = copy.deepcopy(g)
                occupancy_map_un_padded = copy.deepcopy(h)
                occupancy_map_original = copy.deepcopy(p)

                self.grid = Ant44Env0(mj_ant_path=self.mj_ant_path,
                                      re_init=True,
                                      structure=structure,
                                      objects=objects,
                                      obstacle_list=obstacle_list,
                                      occupancy_map=occupancy_map,
                                      default_starts=default_starts,
                                      state_cache=state_cache,
                                      occupancy_map_padded=occupancy_map_padded,
                                      occupancy_map_un_padded=occupancy_map_un_padded,
                                      occupancy_map_original=occupancy_map_original,
                                      waypoint=True)

                start_pos = self.grid.sample_random_pos(number=1)[0]
                goal_pos = self.grid.sample_random_pos(number=1)[0]
                s_goal = self.grid.lls2hls(goal_pos)
                s_init = self.grid.reset(start_pos)

                horizon_left = self.max_iter
                success = False

                path_segment_len_list = self.grid.goal_management.reset(start_pos, goal_pos, self.grid)
                self.grid.old_distance = self.grid.goal_management.path_segment_len_list[0]

                while (horizon_left > 0) and not success:

                    curr_goal = \
                        self.grid.goal_management.way_points[self.grid.goal_management.way_point_current]

                    roll_out, states, actions, rewards, wp_success, l_state = self.roll_out_in_env(
                        start=s_init,
                        goal=curr_goal,
                        ultimate_goal=goal_pos,
                        horizon=horizon_left)

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

                ant_path = self.grid.mj_ant_path + 'ant_copy.xml'
                tree = ET.parse(ant_path)
                tree.write(self.grid.mj_ant_path + 'ant.xml')

            return batch, total_success / j, total_wp_success / jwp, num_steps / j, num_steps / num_roll_outs

        else:

            """ INITIALIZE THE ENVIRONMENT """
            a, b, c, d, e, f, g, h, p = self.grid.reset_env_random()
            structure = copy.deepcopy(a)
            objects = copy.deepcopy(b)
            obstacle_list = copy.deepcopy(c)
            occupancy_map = copy.deepcopy(d)
            default_starts = copy.deepcopy(e)
            state_cache = copy.deepcopy(f)
            occupancy_map_padded = copy.deepcopy(g)
            occupancy_map_un_padded = copy.deepcopy(h)
            occupancy_map_original = copy.deepcopy(p)

            self.grid = Ant44Env0(mj_ant_path=self.mj_ant_path,
                                  re_init=True,
                                  structure=structure,
                                  objects=objects,
                                  obstacle_list=obstacle_list,
                                  occupancy_map=occupancy_map,
                                  default_starts=default_starts,
                                  state_cache=state_cache,
                                  occupancy_map_padded=occupancy_map_padded,
                                  occupancy_map_un_padded=occupancy_map_un_padded,
                                  occupancy_map_original=occupancy_map_original,
                                  waypoint=True)

            start_pos = self.grid.sample_random_pos(number=1)[0]
            goal_pos = self.grid.sample_random_pos(number=1)[0]
            s_goal = self.grid.lls2hls(goal_pos)
            s_init = self.grid.reset(start_pos)

            horizon_left = self.max_iter
            success = False

            path_segment_len_list = self.grid.goal_management.reset(start_pos, goal_pos, self.grid)
            self.grid.old_distance = self.grid.goal_management.path_segment_len_list[0]

            while (horizon_left > 0) and not success:

                curr_goal = \
                    self.grid.goal_management.way_points[self.grid.goal_management.way_point_current]

                roll_out, states, actions, rewards, wp_success, l_state = self.roll_out_in_env(
                    start=s_init,
                    goal=curr_goal,
                    ultimate_goal=goal_pos,
                    horizon=horizon_left)

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

            ant_path = self.grid.mj_ant_path + 'ant_copy.xml'
            tree = ET.parse(ant_path)
            tree.write(self.grid.mj_ant_path + 'ant.xml')

            return success

    def train(self):
        for iter_loop in range(self.inner_episodes):
            batch, g_success, wp_success, steps_trajectory, steps_wp = self.simulate_env(mode='train')
            self.optimizer.process_batch(self.policy, batch, [])
            print(g_success)
            print(wp_success)
            print(steps_trajectory)
            print(steps_wp)

    def test(self):
        g_success = self.simulate_env(mode='test')
        return g_success
