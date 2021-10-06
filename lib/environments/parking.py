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
import numpy as np


class ParkingEmpty:
    def __init__(self):
        self.env = gym.make("parking-v0")
        self.start = (0.0, 0.0)
        self.goal = (0.02, 0.14)
        self.x_size = 24
        self.y_size = 12

        self.occupancy_map = np.zeros((self.x_size, self.y_size))
        self.visitation_map = np.zeros((self.x_size, self.y_size))

        self.occupancy_map_padded = \
            np.logical_not(np.pad(np.array(self.occupancy_map), 1, 'constant', constant_values=1)).astype(int)
        self.occupancy_map_un_padded = np.logical_not(np.array(self.occupancy_map)).astype(int)
        self.occupancy_map_original = copy.deepcopy(self.occupancy_map)

    def reset(self):
        obs = self.env.reset()
        return obs

    def lls2hls(self, state):
        orientation = np.arctan2(state[5], state[4])
        if - np.pi/8 <= orientation < np.pi/8:
            hlo = 0
        elif np.pi/8 <= orientation < 3*np.pi/8:
            hlo = 1
        elif 3*np.pi/8 <= orientation < 5*np.pi/8:
            hlo = 2
        elif 5*np.pi/8 <= orientation < 7*np.pi/8:
            hlo = 3
        elif -7*np.pi/8 <= orientation < -5*np.pi/8:
            hlo = 5
        elif -5*np.pi/8 <= orientation < -3*np.pi/8:
            hlo = 6
        elif -3*np.pi/8 <= orientation < -np.pi/8:
            hlo = 7
        else:
            hlo = 4
        hls = tuple([np.clip(int(25*(state[0]+0.48)), 0, 23), np.clip(int(25*(state[1]+0.24)), 0, 11), hlo])
        return hls

    def hls2lls(self, state):
        return (state[0] / 25) - 0.46, (state[1] / 25) - 0.22

    def check_in_bounds(self, state):
        in_bounds = True
        if (abs(state[0]) > 0.48) or (abs(state[1]) > 0.24):
            in_bounds = False
        return in_bounds
