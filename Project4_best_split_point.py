import numpy as np
from numpy import *
from math import log


def calculate_Gain(feature_list,label,spilt_point):
    # 统计小于等于分割值,不通过类别点的数量
    dic_Y = {'"Iris-versicolor"': 0, '"Iris-setosa"': 0, '"Iris-virginica"': 0}
    dic_N = {'"Iris-versicolor"': 0, '"Iris-setosa"': 0, '"Iris-virginica"': 0}
    nY = 0  # 统计小于等于分割值点的数量
    nN = 0  # 统计大于分割值点的数量
    # 遍历数据，统计各类点的数量
    for i in range(0, len(feature_list)):
        if feature_list[i] <= spilt_point:
            nY += 1
            dic_Y[label[i]] += 1
        else:
            nN += 1
            dic_N[label[i]] += 1

    nY1 = dic_Y['"Iris-versicolor"']  # 统计小于等于分割值,类别属于Iris-versicolor点的数量
    nY2 = dic_Y['"Iris-setosa"']  # 统计小于等于分割值,类别属于Iris-setosa点的数量
    nY3 = dic_Y['"Iris-virginica"']  # 统计小于等于分割值,类别属于Iris-virginica点的数量
    nN1 = dic_N['"Iris-versicolor"']  # 统计大于分割值,类别属于Iris-versicolor点的数量
    nN2 = dic_N['"Iris-setosa"']  # 统计大于分割值,类别属于Iris-setosa点的数量
    nN3 = dic_N['"Iris-virginica"']  # 统计大于分割值,类别属于Iris-virginica点的数量

    # 计算H(D)
    dic = {}
    for curlabel in label:
        if curlabel not in dic.keys():
            dic[curlabel] = 1
        else:
            dic[curlabel] += 1
    HD = 0
    for key in dic:
        # 计算各类别出现的概率
        pxi = float(dic[key]) / len(label)
        HD -= pxi * log(pxi, 2)

    DY = 0
    for i in [nY1, nY2, nY3]:
        if i != 0:
            DY -= (i / nY) * log(i / nY, 2)
    DN = 0
    for i in [nN1, nN2, nN3]:
        if i != 0:
            DN -= (i / nN) * log(i / nN, 2)

    Gain = HD - nY / len(label) * DY - nN / len(label) * DN

    return Gain



def get_best_split_point(feature_list,label):
    sort_feature_list = np.sort(feature_list)
    # 取中间值作为分割点
    middle_point_list = []
    for i in range(0,len(sort_feature_list)-1):
        middle_point = (sort_feature_list[i+1]+sort_feature_list[i])/2
        middle_point_list.append(round(middle_point,2))

    # 数组记录每个分割点的信息增益
    Gain_list = []
    # 遍历分割点
    for index in range(0,len(middle_point_list)):
        spilt_point = middle_point_list[index]
        Gain = calculate_Gain(feature_list,label,spilt_point)
        Gain_list.append(Gain)

    # 找到信息增益最大的分割点
    max_gain = 0
    best_spilt_point = 0

    for i in range(0,len(Gain_list)):
        if Gain_list[i]>max_gain :
            max_gain=Gain_list[i]
            best_spilt_point = middle_point_list[i]

    return best_spilt_point,max_gain

