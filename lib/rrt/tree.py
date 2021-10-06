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


import math


# TREE DEFINITION #
class Node:
    def __init__(self, index, position, parent, distance):
        self.position = position
        self.index = index
        self.distance = distance
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)


class Tree:
    def __init__(self, start, goal):
        self.nodes = [Node(index=0, position=start, parent=None, distance=0)]
        self.node_count = 0
        self.goal = goal
        self.goal_reached = False

    def add_node(self, position, parent, distance):
        self.node_count += 1
        self.nodes[parent].add_child(child=self.node_count)
        self.nodes.append(Node(index=self.node_count, position=position, parent=parent,
                               distance=distance))
        return self.node_count

    def nearest_node(self, node):
        distance = []
        for nodes in self.nodes:
            distance.append(math.sqrt((nodes.position[0]-node[0])**2 + (nodes.position[1]-node[1])**2))
        return distance.index(min(distance)), min(distance)

    def get_nodes_within_distance(self, node, threshold):
        distances = []
        nodes_within_dist = []
        idx = 0
        for nodes in self.nodes:
            d = math.sqrt((nodes.position[0] - node[0]) ** 2 + (nodes.position[1] - node[1]) ** 2)
            if d <= threshold:
                nodes_within_dist.append(idx)
                distances.append(d+self.nodes[idx].distance)
            idx += 1
        return nodes_within_dist, distances
