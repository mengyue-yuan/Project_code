import numpy as np
from math import sqrt
np.set_printoptions(suppress=True)

def get_k(x1,x2):
    dot_data = np.dot(x1.T, x2)
    k = pow(dot_data, 2)
    k = round(k, 4)
    return k

data_path = "/Users/yuanmengyue/Desktop/datasets/iris.txt"

data = np.loadtxt(data_path, delimiter=",", usecols=(0, 1, 2, 3))
row_number = np.size(data, axis=0)


# 每个x投射到特征空间
def to_feature_space(vec):
    mul_list = []
    mul_list.append(vec[0] ** 2)
    mul_list.append(vec[1] ** 2)
    mul_list.append(vec[2] ** 2)
    mul_list.append(vec[3] ** 2)
    mul_list.append(sqrt(2) * vec[0] * vec[1])
    mul_list.append(sqrt(2) * vec[0] * vec[2])
    mul_list.append(sqrt(2) * vec[0] * vec[3])
    mul_list.append(sqrt(2) * vec[1] * vec[2])
    mul_list.append(sqrt(2) * vec[1] * vec[3])
    mul_list.append(sqrt(2) * vec[2] * vec[3])
    return mul_list

# 特征空间
new_feature_space = []
for i in range(0,row_number):
    # 计算特征空间中的点
    mul_list = to_feature_space(data[i])
    new_feature_space.append(mul_list)


# 特征空间居中化
col_number = np.size(new_feature_space, axis=1)
for index in range(0,col_number):
    x1_list = []
    for i in new_feature_space:
        x1_list.append(i[index])

    mean1 = np.mean(x1_list)

    for i in range(0, len(new_feature_space)):
        element1 = new_feature_space[i][index] - mean1
        new_feature_space[i][index] = element1

# 特征空间规范化
# 使得特征空间中点的长度为单位长度
for i in range(0,len(new_feature_space)):
    # 计算向量的模
    squared =[x**2 for x in new_feature_space[i]]
    norm = sqrt(sum(squared))

    for j in range(0,len(new_feature_space[i])):
        new_feature_space[i][j] = new_feature_space[i][j]/norm


new_feature_space = np.array(new_feature_space)

# 特征空间中居中和归一化点的成对点积
K_array = []
for i in range(0,row_number):
    list = []
    for j in range(0,row_number):
        list.append(0)
    K_array.append(list)


for i in range(0,row_number):
    for j in range(0,row_number):
        x1 = new_feature_space[i]
        x2 = new_feature_space[j]
        # 计算点积
        dot_data = np.dot(x1.T,x2)

        # 计算结果放入核矩阵
        K_array[i][j] = dot_data


print("--------------------------------------------------")
K_array = np.array(K_array)
print("特征空间中居中和归一化点的成对点积")
print(K_array)
