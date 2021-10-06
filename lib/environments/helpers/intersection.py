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


# CALCULATE EUCLIDEAN DISTANCE BETWEEN TWO POINTS
def calc_point_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# CALCULATE NORMALIZED DIRECTIONAL VECTOR FROM 2 POINTS
def calc_dir_vec_norm(p1, p2):
    dist = calc_point_dist(p1=p1, p2=p2)
    return ((p2[0] - p1[0]) / dist, (p2[1] - p1[1]) / dist), dist


# CALCULATE SINGLE INTERSECTION POINT OF TWO INTERSECTING LINES
def calc_line_intersection_point(p1_l1, p2_l1, p1_l2, p2_l2):
    x1 = p1_l1[0]
    y1 = p1_l1[1]
    x2 = p2_l1[0]
    y2 = p2_l1[1]
    x3 = p1_l2[0]
    y3 = p1_l2[1]
    x4 = p2_l2[0]
    y4 = p2_l2[1]

    if ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)) < 1e-10:
        if abs(x1 - x2) < 1e-10 and abs(x3 - x4) < 1e-10:
            if (abs(y1 - y3) + abs(y2 - y3)) < (abs(y1 - y4) + abs(y2 - y4)):
                return x3, y3
            else:
                return x3, y4
        elif abs(y1 - y2) < 1e-10 and abs(y3 - y4) < 1e-10:
            if (abs(x1 - x3) + abs(x2 - x3)) < (abs(x1 - x4) + abs(x2 - x4)):
                return x3, y3
            else:
                return x4, y3

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

    return px, py


def calc_line_object_border_intersection_point(p1, p2, obj_def, seg_num):
    if seg_num == 0:
        coll_point = calc_line_intersection_point(p1_l1=p1,
                                                  p2_l1=p2,
                                                  p1_l2=(obj_def[0] - 0.5, obj_def[1] - 0.5),
                                                  p2_l2=(obj_def[0] - 0.5 + obj_def[2], obj_def[1] - 0.5))
        normal_vec = (0, 1)
    elif seg_num == 1:
        coll_point = calc_line_intersection_point(p1_l1=p1,
                                                  p2_l1=p2,
                                                  p1_l2=(obj_def[0] - 0.5 + obj_def[2], obj_def[1] - 0.5),
                                                  p2_l2=(obj_def[0] - 0.5 + obj_def[2], obj_def[1] - 0.5 + obj_def[3]))
        normal_vec = (1, 0)
    elif seg_num == 2:
        coll_point = calc_line_intersection_point(p1_l1=p1,
                                                  p2_l1=p2,
                                                  p1_l2=(obj_def[0] - 0.5, obj_def[1] - 0.5),
                                                  p2_l2=(obj_def[0] - 0.5, obj_def[1] - 0.5 + obj_def[3]))
        normal_vec = (1, 0)
    elif seg_num == 3:
        coll_point = calc_line_intersection_point(p1_l1=p1,
                                                  p2_l1=p2,
                                                  p1_l2=(obj_def[0] - 0.5, obj_def[1] - 0.5 + obj_def[3]),
                                                  p2_l2=(obj_def[0] - 0.5 + obj_def[2], obj_def[1] - 0.5 + obj_def[3]))
        normal_vec = (0, 1)
    return coll_point, normal_vec


# GEOMETRIC OBJECT INTERSECTION CHECK #
def check_intersection_line_seg(line_seg_1, line_seg_2):
    p1s = (line_seg_1[0], line_seg_1[1])
    p1e = (line_seg_1[2], line_seg_1[3])

    p2s = (line_seg_2[0], line_seg_2[1])
    p2e = (line_seg_2[2], line_seg_2[3])

    a1 = p1s[1] - p1e[1]
    b1 = p1e[0] - p1s[0]
    c1 = b1 * p1s[1] + a1 * p1s[0]

    a2 = p2s[1] - p2e[1]
    b2 = p2e[0] - p2s[0]
    c2 = b2 * p2s[1] + a2 * p2s[0]

    l1_x_min = min(p1s[0], p1e[0])
    l1_x_max = max(p1s[0], p1e[0])
    l2_x_min = min(p2s[0], p2e[0])
    l2_x_max = max(p2s[0], p2e[0])
    l1_y_min = min(p1s[1], p1e[1])
    l1_y_max = max(p1s[1], p1e[1])
    l2_y_min = min(p2s[1], p2e[1])
    l2_y_max = max(p2s[1], p2e[1])

    det = a1 * b2 - a2 * b1

    if abs(det) < 1e-5:
        if (abs(a1 * p2s[0] + b1 * p2s[1] - c1) < 1e-5) and (abs(a1 * p2e[0] + b1 * p2e[1] - c1) < 1e-5):

            p1_x_in_l2 = l2_x_min - 1e-10 < p1s[0] < l2_x_max + 1e-10
            p2_x_in_l2 = l2_x_min - 1e-10 < p1e[0] < l2_x_max + 1e-10
            p3_x_in_l1 = l1_x_min - 1e-10 < p2s[0] < l1_x_max + 1e-10
            p4_x_in_l1 = l1_x_min - 1e-10 < p2e[0] < l1_x_max + 1e-10
            p1_y_in_l2 = l2_y_min - 1e-10 < p1s[1] < l2_y_max + 1e-10
            p2_y_in_l2 = l2_y_min - 1e-10 < p1e[1] < l2_y_max + 1e-10
            p3_y_in_l1 = l1_y_min - 1e-10 < p2s[1] < l1_y_max + 1e-10
            p4_y_in_l1 = l1_y_min - 1e-10 < p2e[1] < l1_y_max + 1e-10

            if p1_x_in_l2 or p2_x_in_l2 or p3_x_in_l1 or p4_x_in_l1 or p1_y_in_l2 or p2_y_in_l2 or p3_y_in_l1 \
                    or p4_y_in_l1:
                return True, (1, 2, 3)
            else:
                return False, (1, 2, 3)
        else:
            return False, (1, 2, 3)

    x_inter = (b2 * c1 - b1 * c2) / det
    y_inter = (a1 * c2 - a2 * c1) / det

    x_inter_in_l1 = l1_x_min - 1e-10 < x_inter < l1_x_max + 1e-10
    x_inter_in_l2 = l2_x_min - 1e-10 < x_inter < l2_x_max + 1e-10
    y_inter_in_l1 = l1_y_min - 1e-10 < y_inter < l1_y_max + 1e-10
    y_inter_in_l2 = l2_y_min - 1e-10 < y_inter < l2_y_max + 1e-10

    return x_inter_in_l1 and x_inter_in_l2 and y_inter_in_l1 and y_inter_in_l2, \
           (det, l1_x_min, l2_x_min, l1_y_min, l2_y_min, x_inter, y_inter, l1_x_max, l2_x_max, l1_y_max, l2_y_max,
            x_inter_in_l1, x_inter_in_l2, y_inter_in_l1, y_inter_in_l2)


def check_intersection_object(geom_object, line_seg):
    x = geom_object[0] - 0.5
    y = geom_object[1] - 0.5
    dx = geom_object[2]
    dy = geom_object[3]

    inter_1, _ = check_intersection_line_seg((x, y, x + dx, y), line_seg)
    inter_2, _ = check_intersection_line_seg((x + dx, y, x + dx, y + dy), line_seg)
    inter_3, _ = check_intersection_line_seg((x, y, x, y + dy), line_seg)
    inter_4, _ = check_intersection_line_seg((x, y + dy, x + dx, y + dy), line_seg)

    return inter_1 or inter_2 or inter_3 or inter_4, [inter_1, inter_2, inter_3, inter_4]