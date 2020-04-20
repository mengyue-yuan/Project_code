import mean_shift as ms
from numpy import *
import Clustering

def load_points(filename):
    data_path = filename
    data = np.loadtxt(data_path, delimiter=",", usecols=(0,1,2,3))
    return data


def calculate_purity(data_labels,type0_index_list):
    dic = {}
    for i in type0_index_list:
        if data_labels[i] not in dic.keys():
            dic[data_labels[i]] = 1
        else:
            dic[data_labels[i]] += 1

    # 返回出现次数最多的属性
    main_category = max(dic, key=lambda k: dic[k])
    # 计算空间纯度：出现概率最大的类别的概率
    pxi = float(dic[main_category]) / len(type0_index_list)

    return pxi



if __name__ == '__main__':

    data_path = "/Users/yuanmengyue/Desktop/2020春季课程/万林/datasets/iris.txt"
    data_points = loadtxt(data_path, delimiter=",", usecols=(0,1,2,3))
    data_labels = loadtxt(data_path,delimiter = ',',usecols = (4),dtype = str)

    mean_shifter = ms.MeanShift(kernel='multivariate_gaussian')
    mean_shift_result = mean_shifter.MeanShift(data_points, kernel_bandwidth=[0.4,0.4,0.4,0.4])

    attractors_list = []
    attractors_radius_list = []
    R_list = []

    MIN_DENSITY = 0.0018
    for i in range(0,len(mean_shift_result.shifted_points)):
        # 计算每个吸引子的密度
        point_weights = mean_shifter._point_weight(mean_shift_result.shifted_points[i],data_points,
                                                   kernel_bandwidth=[0.4, 0.4, 0.4, 0.4])
        # 若吸引子的密度大于最小密度min_density，加入到吸引子集合
        if point_weights >= MIN_DENSITY:
            attractors_list.append(mean_shift_result.shifted_points[i].tolist())
            # 再将吸引子吸引的点加入到点的集合R（attractor）中
            R_list.append(mean_shift_result.original_points[i].tolist())
            # 计算每个吸引子的邻域半径
            # 吸引子的邻域半径=最后两步的距离和
            attractors_radius_list.append(mean_shift_result.moving_dis_list[i][-1]+
                                          mean_shift_result.moving_dis_list[i][-2])



    # 将计算出的密度吸引子连成簇
    point_grouper = Clustering.PointGrouper()
    # 如果两个密度吸引子之间的距离，小于邻域半径和，即为一簇
    group_assignments,groups,group_points = point_grouper.group_points(attractors_list,R_list,attractors_radius_list)


    # 输出密度吸引子，及其吸引的点集
    for i in range(0,len(groups)):
        print("密度吸引子:",groups[i][0])
        print("所属点集",group_points[i])


    # 统计每个簇的纯度：出现概率最大的类别的概率
    type0_index_list = []
    type1_index_list = []
    type2_index_list = []
    for i in range(0,len(group_assignments)):
        if group_assignments[i]==0:
            type0_index_list.append(i)
        elif group_assignments[i]==1:
            type1_index_list.append(i)
        elif group_assignments[i]==2:
            type2_index_list.append(i)

    purity_list = []
    purity_list.append(calculate_purity(data_labels,type0_index_list))
    purity_list.append(calculate_purity(data_labels, type1_index_list))
    purity_list.append(calculate_purity(data_labels, type2_index_list))

    # 输出簇的数量，以及每一簇的大小、纯度
    print("簇的数量:", len(groups))
    for i in range(0, len(groups)):
        print("第 %d 簇的大小为 %d 纯度为 %f" % (i, len(groups[i]),purity_list[i]))



