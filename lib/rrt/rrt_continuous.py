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
import numpy as np
from lib.rrt.tree import Tree


# RRT CONNECT DEFINITION #
def sample_node(grid, mode='standard'):
    if mode == 'ant':
        pos = grid.sample_random_pos(number=1)[0]
        return (pos[0] / 4, pos[1] / 4)
    else:
        return grid.sample_random_pos(number=1)[0]


def node_distance(t, node1, node2):
    x1 = t.nodes[node1].position[0]
    y1 = t.nodes[node1].position[1]
    x2 = t.nodes[node2].position[0]
    y2 = t.nodes[node2].position[1]
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


def steer(z_nearest, z_rand_local, dist, delta_d_local):
    if dist > delta_d_local:
        dx = (z_rand_local[0] - z_nearest[0]) / (dist / delta_d_local)
        dy = (z_rand_local[1] - z_nearest[1]) / (dist / delta_d_local)
        z_new = (z_nearest[0] + dx, z_nearest[1] + dy)
        dist_new = math.sqrt((z_new[0]-z_nearest[0])**2+(z_new[1]-z_nearest[1])**2)
        return z_new, dist_new
    else:
        return (z_rand_local[0], z_rand_local[1]), dist


def choose_parent(grid, t, nodes_within_dist, node_dist, nearest_idx, z_new, dist, delta_d):
    i_min = nearest_idx
    c_min = t.nodes[nearest_idx].distance + dist
    q = 0
    for element in nodes_within_dist:
        z_new_star, dist_star = steer(t.nodes[element].position, z_new, node_dist[q], delta_d)
        q += 1
        obstacle_free_new = grid.check_collision(
            line_seg=(t.nodes[element].position[0], t.nodes[element].position[1], z_new_star[0], z_new_star[1]))
        if obstacle_free_new[0] and ((np.array(z_new_star) - np.array(z_new)).mean() < 1e-10):
            c_star = t.nodes[element].distance + dist_star
            if c_star < c_min:
                i_min = element
                c_min = c_star
    return i_min, c_min


def reconnect(t, i_new, i_near, new_dist):
    t.nodes[i_near].parent = i_new
    t.nodes[i_near].distance = new_dist


def rewire(grid, t, nodes_within_dist, i_min, i_new, threshold):
    for element in nodes_within_dist:
        if element != i_min:
            z_new_star, dist_star = steer(t.nodes[i_new].position, t.nodes[element].position, node_distance(
                t=t, node1=i_new, node2=element), threshold)
            obstacle_free_new = grid.check_collision(
                line_seg=(t.nodes[i_new].position[0], t.nodes[i_new].position[1], z_new_star[0], z_new_star[1]))
            new_dist = t.nodes[i_new].distance+dist_star
            if obstacle_free_new[0] and ((np.array(z_new_star) - np.array(t.nodes[element].position)).mean() < 1e-10) \
                    and (new_dist < t.nodes[element].distance):
                reconnect(t, i_new, element, new_dist)


def extend(grid, t, z_rand_local, delta_d):
    nearest_idx, dist = t.nearest_node(node=z_rand_local)
    z_nearest = t.nodes[nearest_idx].position
    z_new, dist = steer(z_nearest, z_rand_local, dist, delta_d)
    obstacle_free = grid.check_collision(line_seg=(z_nearest[0], z_nearest[1], z_new[0], z_new[1]))
    if obstacle_free[0]:
        near_nodes, node_dist = t.get_nodes_within_distance(node=z_new, threshold=delta_d)
        parent_idx, parent_dist = choose_parent(grid, t, near_nodes, node_dist, nearest_idx, z_new, dist, delta_d)
        new_idx = t.add_node(position=z_new, parent=parent_idx, distance=parent_dist)
        rewire(grid=grid, t=t, nodes_within_dist=near_nodes, i_min=parent_idx, i_new=new_idx, threshold=delta_d)
        if (abs(t.nodes[new_idx].position[0] - t.goal[0]) < 1e-10) and \
                (abs(t.nodes[new_idx].position[1] - t.goal[1]) < 1e-10):
            t.goal_reached = True
            print("Goal reached!")
        if (abs(z_new[0] - z_rand_local[0]) < 1e-10) and (abs(z_new[1] - z_rand_local[1]) < 1e-10):
            return 2
        else:
            return 1
    else:
        return 0


def advance(grid, t, z_rand_local, delta_d):
    s = 1
    while s == 1:
        s = extend(grid=grid, t=t, z_rand_local=z_rand_local, delta_d=delta_d)
    return s


def run_rrt(grid_world, max_iter, start_p, goal_p, max_step_size, mode='standard'):
    myTree1 = Tree(start=start_p, goal=goal_p)
    myTree2 = Tree(start=goal_p, goal=start_p)
    full_path = []
    full_path_rev = []
    for iter_loop in range(max_iter):
        z_rand = sample_node(grid=grid_world, mode=mode)
        if iter_loop % 2 == 0:
            tree = myTree1
            other_tree = myTree2
        else:
            tree = myTree2
            other_tree = myTree1
        progress = extend(t=tree, z_rand_local=z_rand, grid=grid_world, delta_d=max_step_size)
        if progress != 0:
            if advance(grid=grid_world, t=other_tree, z_rand_local=tree.nodes[tree.node_count].position,
                       delta_d=max_step_size) == 2:
                path1 = []
                path2 = []
                path_idx1 = myTree1.node_count
                path_idx2 = myTree2.nodes[myTree2.node_count].parent
                while path_idx1 > 0:
                    path1.append(path_idx1)
                    path_idx1 = myTree1.nodes[path_idx1].parent
                path1.append(0)
                while path_idx2 > 0:
                    path2.append(path_idx2)
                    path_idx2 = myTree2.nodes[path_idx2].parent
                path2.append(0)
                for j in reversed(path2):
                    full_path.append(myTree2.nodes[j].position)
                for i in path1:
                    full_path.append(myTree1.nodes[i].position)
                rem_list = []
                for i in range(len(full_path) - 1):
                    if (abs(full_path[i][0] - full_path[i + 1][0]) < 1e-10) and \
                            (abs(full_path[i][1] - full_path[i + 1][1]) < 1e-10):
                        rem_list.append(full_path[i])
                for element in rem_list:
                    full_path.remove(element)
                for i in reversed(range(len(full_path))):
                    full_path_rev.append(full_path[i])
                return full_path_rev, True, myTree1, myTree2
    return full_path_rev, False, myTree1, myTree2


def extract_rrt_states(path_list):
    rrt_states = []
    for path in path_list:
        for idx in range(len(path)):
            if path[idx] not in rrt_states:
                rrt_states.append(path[idx])
    return rrt_states
