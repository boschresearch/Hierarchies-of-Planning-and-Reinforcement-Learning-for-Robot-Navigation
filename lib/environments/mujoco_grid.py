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
import gym
import random
import numpy as np

from lib.rrt.rrt_continuous import run_rrt
from lib.environments.helpers.intersection_mujoco import *
from lib.environments.maze_generation.maze_generation import objects_from_occupancy

import xml.etree.ElementTree as ET


class GoalManagementMonitor:

    def __init__(self):
        self.start = None
        self.goal = None
        self.way_points_coarse = []
        self.way_points = []
        self.num_way_points = 0
        self.way_point_current = 0
        self.target_vec = None
        self.path_segment_len_list = None
        self.remaining_path_len_list = None
        self.waypoint_avail = None

    def reset(self, start, goal, grid):
        self.start = (start[0] / 4, start[1] / 4)
        self.goal = (goal[0] / 4, goal[1] / 4)

        path, rrt_success, _, _ = run_rrt(grid_world=grid,
                                          max_iter=1000,
                                          start_p=self.start,
                                          goal_p=self.goal,
                                          max_step_size=0.25,
                                          mode='ant')

        self.way_points_coarse = copy.deepcopy(path)
        path = [(element[0] * 4, element[1] * 4) for element in path]

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

        if success:
            self.waypoint_avail[self.way_point_current] = 0
            if np.sum(self.waypoint_avail) > 0:
                self.way_point_current += 1
            grid.old_distance = np.linalg.norm(np.array(self.way_points[self.way_point_current]) - np.array(curr_pos))
            return True
        else:
            return False

    def get_target_vec(self, state):
        self.target_vec = np.array(self.way_points[self.way_point_current]) - state
        return self.target_vec


class Ant44Env0:
    def __init__(self, mj_ant_path,
                 waypoint=False,
                 re_init=False,
                 structure=None,
                 objects=None,
                 obstacle_list=None,
                 occupancy_map=None,
                 default_starts=None,
                 state_cache=None,
                 occupancy_map_padded=None,
                 occupancy_map_un_padded=None,
                 occupancy_map_original=None
                 ):
        self.env = gym.make('Ant-v2')
        self.start = (6.0, 18.0)
        self.goal = (14.0, 18.0)
        self.x_size = 24
        self.y_size = 24
        self.mj_ant_path = mj_ant_path

        self.visitation_map = np.zeros((self.x_size, self.y_size))
        self.rp_map = np.zeros((self.x_size, self.y_size))
        self.iter_count = 0

        if not re_init:

            # The following snippet is derived from rllab-curriculum
            #   (https://github.com/florensacc/rllab-curriculum)
            # Copyright (c) 2016 rllab contributors, licensed under the MIT license,
            # cf. 3rd-party-licenses.txt file in the root directory of this source tree.

            ant_path = self.mj_ant_path + 'ant.xml'
            tree = ET.parse(ant_path)
            tree.write(self.mj_ant_path + 'ant_copy.xml')

            ant_path = self.mj_ant_path + 'ant_copy.xml'
            tree = ET.parse(ant_path)
            worldbody = tree.find(".//worldbody")

            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 1, 1],
                [1, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]

            for i in range(len(structure)):
                for j in range(len(structure[0])):
                    if str(structure[i][j]) == '1':
                        # offset all coordinates so that coordinates start at 0
                        ET.SubElement(
                            worldbody, "geom",
                            name="block_%d_%d" % (i, j),
                            pos="%f %f %f" % (i * 4 + 2,
                                              j * 4 + 2,
                                              2),
                            size="%f %f %f" % (2,
                                               2,
                                               2),
                            type="box",
                            material="",
                            contype="1",
                            conaffinity="1",
                            rgba="0.4 0.4 0.4 0.5"
                        )

            root = tree.getroot()
            root[1].attrib["timestep"] = "0.02"

            tree.write(self.mj_ant_path + 'ant.xml')

            # End of snippet.

            self.structure = np.array(structure)
            self.obstacle_list = []
            for i in range(len(structure)):
                for j in range(len(structure[0])):
                    if str(structure[i][j]) == '1':
                        self.obstacle_list.append((i * 4 + 2, j * 4 + 2))

            self.occupancy_map = np.zeros((self.x_size, self.y_size))
            for i in range(6):
                for j in range(6):
                    if structure[i][j] == 1:
                        for inner_i in range(i * 4, min(i * 4 + 4, 24)):
                            for inner_j in range(j * 4, min(j * 4 + 4, 24)):
                                self.occupancy_map[inner_i, inner_j] = 1
            # print(self.occupancy_map)

            self.default_starts = []
            self.state_cache = np.zeros((self.x_size, self.y_size, 8))
            for x_coord in range(self.x_size):
                for y_coord in range(self.y_size):
                    if self.occupancy_map[x_coord][y_coord] == 0:
                        self.default_starts.append((x_coord, y_coord))
                        self.state_cache[x_coord, y_coord, :] = list(self.get_neighborhood((x_coord, y_coord)))

            # print(self.default_starts)

            self.occupancy_map_padded = \
                np.logical_not(np.pad(np.array(self.occupancy_map), 1, 'constant', constant_values=1)).astype(int)
            self.occupancy_map_un_padded = \
                np.logical_not(np.array(self.occupancy_map)).astype(int)
            self.occupancy_map_original = copy.deepcopy(self.occupancy_map)

            self.objects = objects_from_occupancy(self.structure)
            for obj_idx in range(len(self.objects)):
                at = self.objects[obj_idx][0] - 0.2
                bt = self.objects[obj_idx][1] - 0.2
                ct = self.objects[obj_idx][2] + 0.4
                dt = self.objects[obj_idx][3] + 0.4
                self.objects[obj_idx] = (at, bt, ct, dt)

            # print(self.objects)
        else:
            self.structure = structure
            self.objects = objects
            self.obstacle_list = obstacle_list
            self.occupancy_map = occupancy_map
            self.default_starts = default_starts
            self.state_cache = state_cache
            self.occupancy_map_padded = occupancy_map_padded
            self.occupancy_map_un_padded = occupancy_map_un_padded
            self.occupancy_map_original = occupancy_map_original

        if waypoint:
            self.goal_management = GoalManagementMonitor()
            self.old_distance = 0.

    def lls2hls(self, state, robust=False, old_state=None):
        hls = tuple([int(np.clip(state[0], 0, 23)), int(np.clip(state[1], 0, 23))])
        if robust:
            # if self.occupancy_map[hls[0], hls[1]]
            pass
        else:
            return hls

    def lls2hls4(self, state):
        return tuple([np.clip(int(state[0] / 4), 0, 5), np.clip(int(state[1] / 4), 0, 5)])

    def hls2lls(self, state):
        return float(state[0]), float(state[1])

    def reset_env_random(self, set_grid=False, grid_type=0):
        self.start = None
        self.goal = None
        self.x_size = 24
        self.y_size = 24

        if not set_grid:
            grid_type = random.randint(0, 24)
        if grid_type == 0:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 1, 1],
                [1, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 1:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 2:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 3:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 4:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 5:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 1, 1],
                [1, 1, 1, 0, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 6:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 7:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 8:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 9:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 0, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 10:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 0, 1],
                [1, 1, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 11:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 12:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 13:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 14:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 15:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 0, 0, 1, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 16:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 17:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 18:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 1, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 19:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 0, 1],
                [1, 1, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 20:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 21:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 22:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 23:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        elif grid_type == 24:
            structure = [
                [1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1],
                [1, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]

        # The following snippet is derived from rllab-curriculum
        #   (https://github.com/florensacc/rllab-curriculum)
        # Copyright (c) 2016 rllab contributors, licensed under the MIT license,
        # cf. 3rd-party-licenses.txt file in the root directory of this source tree.

        ant_path = self.mj_ant_path + 'ant_copy.xml'
        tree = ET.parse(ant_path)
        worldbody = tree.find(".//worldbody")

        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if str(structure[i][j]) == '1':
                    # offset all coordinates so that coordinates start at 0
                    ET.SubElement(
                        worldbody, "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f" % (i * 4 + 2,
                                          j * 4 + 2,
                                          2),
                        size="%f %f %f" % (2,
                                           2,
                                           2),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.4 0.4 0.4 0.5"
                    )

        root = tree.getroot()
        root[1].attrib["timestep"] = "0.02"

        tree.write(self.mj_ant_path + 'ant.xml')

        # End of snippet.

        self.structure = np.array(structure)
        self.objects = objects_from_occupancy(self.structure)
        for obj_idx in range(len(self.objects)):
            at = self.objects[obj_idx][0] - 0.2
            bt = self.objects[obj_idx][1] - 0.2
            ct = self.objects[obj_idx][2] + 0.4
            dt = self.objects[obj_idx][3] + 0.4
            self.objects[obj_idx] = (at, bt, ct, dt)

        self.obstacle_list = []
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if str(structure[i][j]) == '1':
                    self.obstacle_list.append((i * 4 + 2, j * 4 + 2))

        self.occupancy_map = np.zeros((self.x_size, self.y_size))
        for i in range(6):
            for j in range(6):
                if structure[i][j] == 1:
                    for inner_i in range(i * 4, min(i * 4 + 4, 24)):
                        for inner_j in range(j * 4, min(j * 4 + 4, 24)):
                            self.occupancy_map[inner_i, inner_j] = 1
        # print(self.occupancy_map)

        self.visitation_map = np.zeros((self.x_size, self.y_size))
        self.rp_map = np.zeros((self.x_size, self.y_size))
        self.iter_count = 0

        self.default_starts = []
        self.state_cache = np.zeros((self.x_size, self.y_size, 8))
        for x_coord in range(self.x_size):
            for y_coord in range(self.y_size):
                if self.occupancy_map[x_coord][y_coord] == 0:
                    self.default_starts.append((x_coord, y_coord))
                    self.state_cache[x_coord, y_coord, :] = list(self.get_neighborhood((x_coord, y_coord)))

        # print(self.default_starts)

        self.occupancy_map_padded = \
            np.logical_not(np.pad(np.array(self.occupancy_map), 1, 'constant', constant_values=1)).astype(int)
        self.occupancy_map_un_padded = \
            np.logical_not(np.array(self.occupancy_map)).astype(int)
        self.occupancy_map_original = copy.deepcopy(self.occupancy_map)

        return self.structure, self.objects, self.obstacle_list, self.occupancy_map, self.default_starts, \
               self.state_cache, self.occupancy_map_padded, self.occupancy_map_un_padded, self.occupancy_map_original

    def observation(self):
        return np.concatenate([self.env.sim.data.qpos.flat, self.env.sim.data.qvel.flat, ]).reshape(-1)

    def reset(self, start):
        self.env.reset()
        qpos = self.env.data.qpos
        qpos[0] = start[0]
        qpos[1] = start[1]
        qvel = self.env.data.qvel
        self.env.set_state(qpos, qvel)
        return self.observation()

    def step(self, a, goal):
        _, _, _, _ = self.env.step(action=a)
        s_new = self.observation()
        reward = 0
        done = False
        goal_dist = np.linalg.norm(s_new[:2] - goal)
        if goal_dist < 0.5:
            reward = 1.0
            done = True
        return s_new, reward, done, dict(no_goal_reached=not done)

    def sample_random_pos(self, number):
        start_list = []
        while len(start_list) < number:
            x_coord = random.uniform(0.0, 24.0)
            y_coord = random.uniform(0.0, 24.0)
            admissible = True
            for obstacle in self.obstacle_list:
                if obstacle[0] - 2.75 < x_coord < obstacle[0] + 2.75 and \
                        obstacle[1] - 2.75 < y_coord < obstacle[1] + 2.75:
                    admissible = False
            if admissible:
                start_list.append((x_coord, y_coord))
        return start_list

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

    def get_occupancy(self, s_in):
        if 0 <= s_in[0] < self.x_size and 0 <= s_in[1] < self.y_size:
            return self.occupancy_map[s_in[0], s_in[1]]
        else:
            return 1

    def get_neighborhood(self, state):

        return self.get_occupancy(s_in=(state[0]-1, state[1]+1)), self.get_occupancy(s_in=(state[0], state[1]+1)), \
               self.get_occupancy(s_in=(state[0]+1, state[1]+1)), self.get_occupancy(s_in=(state[0]-1, state[1])), \
               self.get_occupancy(s_in=(state[0]+1, state[1])), self.get_occupancy(s_in=(state[0]-1, state[1]-1)), \
               self.get_occupancy(s_in=(state[0], state[1]-1)), self.get_occupancy(s_in=(state[0]+1, state[1]-1))
