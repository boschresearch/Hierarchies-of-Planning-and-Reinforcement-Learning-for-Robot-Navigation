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


def objects_from_occupancy(m):
    # Object List for Occupancy
    o = list()
    for i in range(m.shape[0]):
        x = i
        dx = 1
        dy = 0
        ob = False
        for j in range(m.shape[1]):
            if m[i, j] and not ob:
                ob = True
                y = copy.deepcopy(j)
                dy = 1
                if j == (m.shape[1] - 1):
                    ob = False
                    o.append((x, y, dx, dy))
            elif m[i, j] and ob:
                dy += 1
                if j == (m.shape[1] - 1):
                    ob = False
                    o.append((x, y, dx, dy))
            elif (not m[i, j]) and ob:
                ob = False
                o.append((x, y, dx, dy))
    return o
