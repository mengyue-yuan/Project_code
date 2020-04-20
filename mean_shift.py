import math
import numpy as np

# 两点间的容差
MIN_DISTANCE = 0.0001


def euclidean_dist(pointA, pointB):
    if(len(pointA) != len(pointB)):
        raise Exception("expected point dimensionality to match")
    total = float(0)
    for dimension in range(0, len(pointA)):
        total += (pointA[dimension] - pointB[dimension])**2
    return math.sqrt(total)


def gaussian_kernel(distance, bandwidth):
    euclidean_distance = np.sqrt(((distance)**2).sum(axis=1))
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((euclidean_distance / bandwidth))**2)
    return val


def multivariate_gaussian_kernel(distances, bandwidths):

    # Number of dimensions of the multivariate gaussian
    dim = len(bandwidths)
    # 协方差矩阵
    cov = np.multiply(np.power(bandwidths, 2), np.eye(dim))
    # Compute Multivariate gaussian (vectorized implementation)
    exponent = -0.5 * np.sum(np.multiply(np.dot(distances, np.linalg.inv(cov)), distances), axis=1)
    val = (1 / np.power((2 * math.pi), (dim/2)) * np.power(np.linalg.det(cov), 0.5)) * np.exp(exponent)
    return val


class MeanShift(object):
    def __init__(self, kernel=gaussian_kernel):
        if kernel == 'multivariate_gaussian':
            kernel = multivariate_gaussian_kernel
        self.kernel = kernel

    def MeanShift(self, points, kernel_bandwidth, iteration_callback=None):
        if(iteration_callback):
            iteration_callback(points, 0)
        # shift_points数组存放点移动后的位置
        shift_points = np.array(points)
        # moving_dis_list数组存放每个点每次移动的距离
        # 取最后两个距离之和作为密度吸引子的邻域半径
        moving_dis_list = []
        for i in range(0,len(points)):
            moving_dis_list.append([])

        max_min_dist = 1
        iteration_number = 0
        # 记录点是否已经完成迭代，初始化为Ture，迭代完成改为False
        still_shifting = [True] * points.shape[0]
        # 当所有的点都完成迭代时， max_min_dist=0退出while循环
        while max_min_dist > MIN_DISTANCE:
            max_min_dist = 0
            iteration_number += 1
            for i in range(0, len(shift_points)):
                # 判断每个样本点是否需要继续迭代
                if not still_shifting[i]:
                    continue
                p_new = shift_points[i]
                p_new_start = p_new
                # 更新漂移后的点
                p_new = self._shift_point(p_new, points, kernel_bandwidth)
                # 计算原始点到更新点的移动距离
                dist = euclidean_dist(p_new, p_new_start)
                moving_dis_list[i].append(dist)
                # 记录点移动的较大距离
                if dist > max_min_dist:
                    max_min_dist = dist
                # 距离小于容差，此时点已经不需要移动
                if dist < MIN_DISTANCE:
                    still_shifting[i] = False
                # 漂移后的点存入shift_points数组
                shift_points[i] = p_new

            if iteration_callback:
                iteration_callback(shift_points, iteration_number)

        return MeanShiftResult(points, shift_points,moving_dis_list)



    def _point_weight(self, point, points, kernel_bandwidth):
        # 通过高斯核计算每个点的权重
        point_weights = self.kernel(point - points, kernel_bandwidth)
        density = sum(point_weights)/150
        # 超立方体的体积
        V = 1
        for i in kernel_bandwidth:
            V = V *i
        return density/V




    # 返回一次更新后的点
    def _shift_point(self, point, points, kernel_bandwidth):
        points = np.array(points)
        # 通过高斯核计算每个点的权重
        point_weights = self.kernel(point-points, kernel_bandwidth)
        tiled_weights = np.tile(point_weights, [len(point), 1])
        # 计算分母
        denominator = sum(point_weights)
        # 计算更新点
        shifted_point = np.multiply(tiled_weights.transpose(),points).sum(axis=0)/denominator
        return shifted_point



class MeanShiftResult:
    def __init__(self, original_points, shifted_points,moving_dis_list):
        self.original_points = original_points
        self.shifted_points = shifted_points
        # self.cluster_ids = cluster_ids
        self.moving_dis_list = moving_dis_list
