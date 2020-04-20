import sys
import numpy as np
from mean_shift import euclidean_dist



class Cluster(object):
    def group_points(self, points,R_list,radius_list):
        # 记录每个点的所属类别
        group_assignment = []
        # 记录每个类别中吸引子的集合
        groups = []
        # 记录每个类别中点的集合
        group_points = []
        # 记录每个类别的邻域半径
        groups_radius = []
        group_index = 0
        for i in range(0,len(points)):
            # 计算point所属簇
            nearest_group_index = self._determine_nearest_group(points[i],radius_list[i], groups,groups_radius)

            if nearest_group_index is None:
                # 如果集合中还没有该簇，则新建簇
                groups.append([points[i]])
                group_points.append([R_list[i]])
                groups_radius.append([radius_list[i]])
                group_assignment.append(group_index)
                group_index += 1
            else:
                # 加入该簇集合
                group_assignment.append(nearest_group_index)
                groups[nearest_group_index].append(points[i])
                group_points[nearest_group_index].append(R_list[i])
                groups_radius[nearest_group_index].append(radius_list[i])
        return np.array(group_assignment),groups,group_points


    # 计算point所属簇
    def _determine_nearest_group(self, point,radius,groups,groups_radius):
        nearest_group_index = None
        index = 0
        for i in range(0,len(groups)):
            # 返回与簇中点的最小距离和簇内点的邻域半径
            distance_to_group,group_radius = self._distance_to_group(point, groups[i],groups_radius[i])
            # 如果两个密度吸引子之间的距离，小于邻域半径和，即为一簇
            if distance_to_group < (radius+group_radius):
                nearest_group_index = index
            index += 1
        return nearest_group_index



    # 返回与簇中点的最小距离和簇内点的邻域半径
    def _distance_to_group(self, point, group,radius_list):
        min_distance = sys.float_info.max
        for i in range(0,len(group)):
            dist = euclidean_dist(point, group[i])
            if dist < min_distance:
                min_distance = dist
                radius = radius_list[i]

        return min_distance,radius
