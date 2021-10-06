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


import random
import numpy as np
import copy

from lib.environments.helpers.intersection import *
from lib.rrt.rrt_continuous import run_rrt
from lib.environments.maze_generation.maze_generation import objects_from_occupancy


# WAYPOINT GOAL MANAGER
class GoalManagementMonitor:

    def __init__(self):
        self.start = None
        self.goal = None
        self.way_points = []
        self.num_way_points = 0
        self.way_point_current = 0
        self.target_vec = None
        self.path_segment_len_list = None
        self.remaining_path_len_list = None
        self.waypoint_avail = None

    def reset(self, start, goal, grid):
        self.start = start
        self.goal = goal

        path, rrt_success, _, _ = run_rrt(grid_world=grid,
                                          max_iter=2500,
                                          start_p=start,
                                          goal_p=goal,
                                          max_step_size=2)

        if not rrt_success:
            path = [start, goal]

        self.way_points = path
        self.num_way_points = len(path)
        self.way_point_current = 1
        self.waypoint_avail = np.ones((self.num_way_points, ))
        self.waypoint_avail[0] = 0
        self.target_vec = np.array(self.way_points[self.way_point_current]) - \
                          np.array(self.way_points[self.way_point_current - 1])

        self.path_segment_len_list = np.zeros((self.num_way_points - 1, ))
        for w in range(1, self.num_way_points):
            wp_dist = np.linalg.norm(np.array(self.way_points[w]) - np.array(self.way_points[w - 1]))
            self.path_segment_len_list[w - 1] = wp_dist
        self.remaining_path_len_list = np.zeros((self.num_way_points, ))
        for idx in reversed(range(self.num_way_points - 1)):
            self.remaining_path_len_list[idx] = self.remaining_path_len_list[idx + 1] + self.path_segment_len_list[idx]

        return self.path_segment_len_list

    def update_way_point(self, grid, curr_pos, success):
        if np.sum(self.waypoint_avail) > 1:
            """
            if success:
                self.waypoint_avail[self.way_point_current] = 0
            """
            changed = False
            dist_list = []
            wp_candid = []
            for wpi in range(self.num_way_points):
                if self.waypoint_avail[wpi]:
                    dist_list.append(np.linalg.norm(np.array(self.way_points[wpi]) - np.array(curr_pos)) +
                                     self.remaining_path_len_list[wpi])
                    wp_candid.append(wpi)
            shortest = 1000
            shortest_idx = self.way_point_current
            for dl in range(len(dist_list)):
                obstacle_free = grid.check_collision(line_seg=(self.way_points[wp_candid[dl]][0],
                                                               self.way_points[wp_candid[dl]][1],
                                                               curr_pos[0],
                                                               curr_pos[1])
                                                     )
                if obstacle_free[0] and (dist_list[dl] < shortest):
                    shortest = dist_list[dl]
                    shortest_idx = wp_candid[dl]
            wp_idx_proposal = int(shortest_idx)
            if wp_idx_proposal != self.way_point_current:
                changed = True
            self.way_point_current = wp_idx_proposal
            grid.old_distance = np.linalg.norm(np.array(self.way_points[wp_idx_proposal]) - np.array(curr_pos))

            return changed
        else:
            return False

    def get_target_vec(self, state):
        self.target_vec = np.array(self.way_points[self.way_point_current]) - state
        return self.target_vec


# GRID WORLD DEFINITION #
class GridWorldContinuous:

    def __init__(self,
                 grid_mode='standard',
                 layout=None,
                 objects=None,
                 delta_t=0.1,
                 col_break=False,
                 goal_rew=1,
                 col_rew=0,
                 time_rew=0,
                 a_lim=100,
                 v_lim=10,
                 usym=False,
                 waypoint=False,
                 terrain=False,
                 ):

        def add_object(object_params):
            for i_g in range(object_params[0], object_params[0] + object_params[2]):
                for j_g in range(object_params[1], object_params[1] + object_params[3]):
                    self.occupancy_map[i_g][j_g] = 1

        self.state = None
        self.start = None
        self.goal = None

        if grid_mode == 'standard':
            self.x_size = 30
            self.y_size = 20
            self.objects = [(3, 3, 1, 4),
                            (4, 6, 3, 1),
                            (7, 6, 1, 3),
                            (7, 0, 1, 3),
                            (8, 2, 3, 1),
                            (10, 3, 1, 3),
                            (14, 5, 1, 7),
                            (13, 2, 3, 1),
                            (14, 0, 1, 2),
                            (19, 2, 1, 3),
                            (20, 4, 5, 1),
                            (23, 0, 1, 4),
                            (27, 0, 1, 5),
                            (24, 5, 1, 3),
                            (21, 6, 1, 1),
                            (18, 4, 1, 7),
                            (15, 11, 4, 1),
                            (8, 8, 6, 1),
                            (4, 10, 2, 1),
                            (5, 11, 9, 1),
                            (3, 14, 1, 6),
                            (23, 8, 4, 1),
                            (26, 9, 1, 3),
                            (22, 8, 1, 7),
                            (27, 15, 3, 1),
                            (26, 15, 1, 2),
                            (7, 15, 16, 1),
                            (16, 16, 1, 2),
                            (10, 18, 1, 2)]
            self.occupancy_map = np.zeros((self.x_size, self.y_size))
            for item in self.objects:
                add_object(item)
            self.goal = (25., 1.)
        else:
            self.x_size = layout.shape[0]
            self.y_size = layout.shape[1]
            self.occupancy_map = layout
            self.objects = objects

        self.col_break = col_break
        self.usym = usym
        self.terrain = terrain
        self.waypoint = waypoint

        self.dt = delta_t
        self.iter_count = 0

        self.a_lim = a_lim
        self.v_lim = v_lim

        self.goal_rew = goal_rew
        self.col_rew = col_rew
        self.time_rew = time_rew

        self.visitation_map = np.zeros((self.x_size, self.y_size))
        self.terrain_map = copy.deepcopy(self.occupancy_map)

        self.state_cache = np.zeros((self.x_size, self.y_size, 8))
        self.default_starts = []
        for x_coord in range(self.x_size):
            for y_coord in range(self.y_size):
                if self.occupancy_map[x_coord][y_coord] == 0:
                    self.default_starts.append((x_coord, y_coord))
        for start in self.default_starts:
            self.state_cache[start[0], start[1], :] = list(self.get_neighborhood(start))

        print("ENV INITIALIZED")
        print(self.occupancy_map)
        # np.save("occupancy_map", np.array(self.occupancy_map))
        self.occupancy_map_padded = \
            np.logical_not(np.pad(np.array(self.occupancy_map), 1, 'constant', constant_values=1)).astype(int)
        self.occupancy_map_un_padded = \
            np.logical_not(np.array(self.occupancy_map)).astype(int)
        self.terrain_map_un_padded = copy.deepcopy(self.occupancy_map_un_padded)

        if waypoint:
            self.goal_management = GoalManagementMonitor()
            self.old_distance = 0.

    def check_occupancy(self, position):
        if self.occupancy_map[min(max(0, int(round(position[0]))), self.x_size)][min(max(0, int(round(position[1]))),
                                                                                     self.y_size)] == 0:
            return False
        else:
            return True

    def lls2hls(self, state):
        hls = (max(min(int(round(state[0])), self.x_size - 1), 0),
               max(min(int(round(state[1])), self.y_size - 1), 0))
        return hls

    def lls2hls_false(self, state):
        hls = (max(min(int(state[0]), self.x_size - 1), 0),
               max(min(int(state[1]), self.y_size - 1), 0))
        return hls

    def hls2lls(self, state):
        return float(state[0]), float(state[1])

    def reset(self, start_p, goal_p):
        self.iter_count = 0
        self.start = start_p
        self.goal = goal_p
        # self.visitation_map = np.zeros((self.x_size, self.y_size))
        self.state = np.array(list(self.start) + [0, 0])
        return self.state

    def reset_env_terrain(self):
        z = np.zeros((21, 21))
        z[0, :] = 1
        z[20, :] = 1
        z[:, 0] = 1
        z[:, 20] = 1
        z[12, 1:6] = 1
        z[12, 7:13] = 1
        z[8, 12:16] = 1
        z[8, 17:20] = 1
        z[1:4, 12] = 1
        z[5:16, 12] = 1
        z[17:20, 12] = 1
        o = objects_from_occupancy(z)
        self.occupancy_map = z
        self.objects = np.array(o)
        self.x_size = 21
        self.y_size = 21

        self.state = None
        self.start = None
        self.goal = None

        self.visitation_map = np.zeros((self.x_size, self.y_size))
        self.terrain_map = copy.deepcopy(self.occupancy_map).astype(np.float)

        self.default_starts = []
        for x_coord in range(self.x_size):
            for y_coord in range(self.y_size):
                if self.occupancy_map[x_coord][y_coord] == 0:
                    self.default_starts.append((x_coord, y_coord))

        self.occupancy_map_un_padded = np.logical_not(np.array(self.occupancy_map)).astype(int)
        self.terrain_map_un_padded = copy.deepcopy(self.occupancy_map_un_padded).astype(np.float)
        if self.terrain:
            for tx in [1, 2, 3, 4, 5, 6, 7]:
                for ty in [13, 14, 15, 16, 17, 18, 19]:
                    self.terrain_map_un_padded[tx, ty] = 0.5
                    self.terrain_map[tx, ty] = 0.5
            self.terrain_map[4, 12] = 0.5
            self.terrain_map[8, 16] = 0.5
            self.terrain_map_un_padded[4, 12] = 0.5
            self.terrain_map_un_padded[8, 16] = 0.5

        self.state_cache = np.zeros((self.x_size, self.y_size, 8))
        for start in self.default_starts:
            self.state_cache[start[0], start[1], :] = list(self.get_neighborhood(start))
        self.occupancy_map_padded = \
            np.logical_not(np.pad(np.array(self.occupancy_map), 1, 'constant', constant_values=1)).astype(int)

    def step(self, action, eps_val=1e-3):

        pos_list = []

        action = np.clip(action, a_min=-self.a_lim, a_max=self.a_lim)

        if self.usym:
            if action[0] > 0:
                a0 = 1.
            else:
                a0 = 0.25
            if action[1] > 0:
                a1 = 1.
            else:
                a1 = 0.25
            self.state[2:4] = np.clip(self.state[2:4] + action * np.array([a0, a1]) * self.dt, a_min=-self.v_lim,
                                      a_max=self.v_lim)
        elif self.terrain:
            hls = self.lls2hls(self.state)
            if 0.49 < self.terrain_map[hls[0], hls[1]] < 0.51:
                self.state[2:4] = np.clip(self.state[2:4] + action * 0.1 * self.dt, a_min=-self.v_lim, a_max=self.v_lim)
            else:
                self.state[2:4] = np.clip(self.state[2:4] + action * self.dt, a_min=-self.v_lim, a_max=self.v_lim)
        else:
            self.state[2:4] = np.clip(self.state[2:4] + action * self.dt, a_min=-self.v_lim, a_max=self.v_lim)

        p_new = self.state[0:2] + self.state[2:4] * self.dt

        pos_list.append(tuple(self.state[0:2]))

        collision_free, causing_obj, causing_seg_list = \
            self.check_collision(line_seg=(self.state[0], self.state[1], p_new[0], p_new[1]))
        collision_var_tot = \
            (not (-0.5 < p_new[0] < self.x_size - 0.5)) or \
            (not (-0.5 < p_new[1] < self.y_size - 0.5)) or \
            not collision_free
        collision_indicator = collision_var_tot

        if collision_var_tot:
            if collision_var_tot:
                collision_not_found = False
                if not collision_free:
                    coll_tuples = []
                    for obj_i in range(len(causing_obj)):
                        seg_i = 0
                        for seg in causing_seg_list[obj_i]:
                            if seg:
                                coll_tuples.append((causing_obj[obj_i], seg_i))
                            seg_i += 1
                    coll_point_list = []
                    for coll_tuple in coll_tuples:
                        coll_point_temp, _ = calc_line_object_border_intersection_point(
                            p1=self.state[0:2], p2=p_new, obj_def=self.objects[coll_tuple[0]], seg_num=coll_tuple[1])
                        coll_point_list.append(coll_point_temp)
                    coll_dist_list = []
                    for coll_point_element in coll_point_list:
                        coll_dist_list.append(calc_point_dist(p1=self.state[0:2], p2=coll_point_element))
                    coll_idx = coll_dist_list.index(min(coll_dist_list))
                    coll_point_final = coll_point_list[coll_idx]
                else:
                    bound_seg_list = [(-0.5, -0.5, -0.5, self.y_size - 0.5),
                                      (self.x_size - 0.5, -0.5, self.x_size - 0.5, self.y_size - 0.5),
                                      (-0.5, self.y_size - 0.5, self.x_size - 0.5, self.y_size - 0.5),
                                      (-0.5, -0.5, self.x_size - 0.5, -0.5)]
                    left_bound_col, save1 = check_intersection_line_seg(
                        line_seg_1=bound_seg_list[0],
                        line_seg_2=(self.state[0], self.state[1], p_new[0], p_new[1]))
                    right_bound_col, save2 = check_intersection_line_seg(
                        line_seg_1=bound_seg_list[1],
                        line_seg_2=(self.state[0], self.state[1], p_new[0], p_new[1]))
                    up_bound_col, save3 = check_intersection_line_seg(
                        line_seg_1=bound_seg_list[2],
                        line_seg_2=(self.state[0], self.state[1], p_new[0], p_new[1]))
                    down_bound_col, save4 = check_intersection_line_seg(
                        line_seg_1=bound_seg_list[3],
                        line_seg_2=(self.state[0], self.state[1], p_new[0], p_new[1]))
                    coll_idx = 0
                    coll_bound_seg = []
                    for bound_col in [left_bound_col, right_bound_col, up_bound_col, down_bound_col]:
                        if bound_col:
                            coll_bound_seg = bound_seg_list[coll_idx]
                        coll_idx += 1
                    if not coll_bound_seg:
                        with open('error_log.txt', 'a') as f:
                            f.write(str(p_new))
                            f.write(str(save1))
                            f.write(str(save2))
                            f.write(str(save3))
                            f.write(str(save4))
                            print("COLLISION NOT FOUND")
                            collision_not_found = True
                    else:
                        coll_point_final = calc_line_intersection_point(p1_l1=self.state[0:2],
                                                                        p2_l1=p_new,
                                                                        p1_l2=coll_bound_seg[0:2],
                                                                        p2_l2=coll_bound_seg[2:4])

                if collision_not_found:
                    self.state[2] = 0
                    self.state[3] = 0
                    self.state[0] = p_new[0]
                    self.state[1] = p_new[1]
                    pos_list.append(tuple(self.state[0:2]))
                    self.state[0] = np.clip(self.state[0], a_min=-0.5, a_max=self.x_size - 0.5)
                    self.state[1] = np.clip(self.state[1], a_min=-0.5, a_max=self.y_size - 0.5)
                else:
                    self.state[2] = 0
                    self.state[3] = 0
                    if False:
                        self.state[0] = self.state[0] + (1 - 2 * eps_val) * (coll_point_final[0] - self.state[0])
                        self.state[1] = self.state[1] + (1 - 2 * eps_val) * (coll_point_final[1] - self.state[1])
                    else:
                        v_norm = np.sqrt(((coll_point_final[0] - self.state[0]) ** 2) +
                                         ((coll_point_final[1] - self.state[1]) ** 2))
                        lambda_val = 1. - (eps_val / v_norm)
                        self.state[0] = self.state[0] + lambda_val * (coll_point_final[0] - self.state[0])
                        self.state[1] = self.state[1] + lambda_val * (coll_point_final[1] - self.state[1])

                    pos_list.append(tuple(self.state[0:2]))
        else:
            self.state[0:2] = p_new

        self.visitation_map[max(min(int(round(self.state[0])), self.x_size - 1), 0),
                            max(min(int(round(self.state[1])), self.y_size - 1), 0)] += 1

        reward = 0
        done_wp = False

        if self.waypoint:
            dist_wp = np.sqrt(
                (self.state[0] - self.goal_management.way_points[self.goal_management.way_point_current][0]) ** 2 +
                (self.state[1] - self.goal_management.way_points[self.goal_management.way_point_current][1]) ** 2)
            done_wp = dist_wp < 0.5
            new_distance = dist_wp
            done = np.sqrt((self.state[0] - self.goal[0]) ** 2 + (self.state[1] - self.goal[1]) ** 2) < 0.5
            reward = done_wp + self.old_distance - 0.99 * new_distance
            self.old_distance = new_distance
        else:
            state_round = np.round(self.state[0:2])
            done = ((state_round[0] == self.goal[0]) and (state_round[1] == self.goal[1]))
            if done:
                reward = self.goal_rew

        if self.col_break:
            break_var = done or collision_indicator
        else:
            break_var = done

        # self.iter_count += 1

        return self.state, reward, done, done_wp, not break_var

    def check_collision(self, line_seg):
        collision_total = False
        obj = 0
        coll_obj = []
        coll_seg_list = []
        for obstacles in self.objects:
            collision, coll_seg = check_intersection_object(geom_object=obstacles, line_seg=line_seg)
            if collision:
                coll_obj.append(obj)
                coll_seg_list.append(coll_seg)
            collision_total = collision_total or collision
            obj += 1
        return not collision_total, coll_obj, coll_seg_list

    def sample_random_pos(self, number):
        start_list = []
        while len(start_list) < number:
            x_coord = random.uniform(-0.5, self.x_size - 0.5)
            y_coord = random.uniform(-0.5, self.y_size - 0.5)
            if self.occupancy_map[round(x_coord)][round(y_coord)] == 0 and (x_coord, y_coord) not in start_list:
                start_list.append((x_coord, y_coord))
        return start_list

    def sample_random_start_terrain(self, number):
        start_list = []
        while len(start_list) < number:
            x_coord = random.uniform(0.5, 11.5)
            y_coord = random.uniform(0.5, 11.5)
            if self.occupancy_map[round(x_coord)][round(y_coord)] == 0 and (x_coord, y_coord) not in start_list:
                start_list.append((x_coord, y_coord))
        return start_list

    def sample_random_goal_terrain(self, number):
        start_list = []
        while len(start_list) < number:
            x_coord = random.uniform(8.5, 12.5)
            y_coord = random.uniform(12.5, 19.5)
            if self.occupancy_map[round(x_coord)][round(y_coord)] == 0 and (x_coord, y_coord) not in start_list:
                start_list.append((x_coord, y_coord))
        return start_list

    def get_occupancy(self, s_in):
        if 0 <= s_in[0] < self.x_size and 0 <= s_in[1] < self.y_size:
            if not self.terrain:
                return self.occupancy_map[s_in[0], s_in[1]]
            else:
                return self.terrain_map[s_in[0], s_in[1]]
        else:
            return 1

    def get_neighborhood(self, state):

        return self.get_occupancy(s_in=(state[0]-1, state[1]+1)), self.get_occupancy(s_in=(state[0], state[1]+1)), \
               self.get_occupancy(s_in=(state[0]+1, state[1]+1)), self.get_occupancy(s_in=(state[0]-1, state[1])), \
               self.get_occupancy(s_in=(state[0]+1, state[1])), self.get_occupancy(s_in=(state[0]-1, state[1]-1)), \
               self.get_occupancy(s_in=(state[0], state[1]-1)), self.get_occupancy(s_in=(state[0]+1, state[1]-1))
