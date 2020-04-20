import numpy as np
from math import sqrt
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

# 读取数据
data_path = "/Users/yuanmengyue/Desktop/datasets/iris.txt"
data = np.loadtxt(data_path, delimiter=",", usecols=(0, 1, 2, 3))
# 数据行数
row_number = np.size(data, axis=0)


# 初始化核矩阵
K_array = []
for i in range(0,row_number):
    list = []
    for j in range(0,row_number):
        list.append(0)
    K_array.append(list)


for i in range(0,row_number):
    for j in range(0,row_number):
        x1 = data[i]
        x2 = data[j]
        # 计算点积
        dot_data = np.dot(x1.T,x2)
        # 平方
        k = dot_data**2
        # 计算结果放入核矩阵
        K_array[i][j] = k


print("--------------------------------------------------")
K_array = np.array(K_array)
print("核矩阵")
print(K_array)
print("--------------------------------------------------")

# 居中化
# 初始化居中核矩阵
KC_array = []
for i in range(0,row_number):
    list = []
    for j in range(0,row_number):
        list.append(0)
    KC_array.append(list)



for i in range(0,row_number):
    for j in range(0,row_number):

        k = K_array[i][j]
        #计算核矩阵中第i行之和
        xi_sum = np.sum(K_array[i])
        # 计算核矩阵中第j列之和
        xj_sum = 0
        for row in range(0,row_number):
            for col in range(0, row_number):
                if col==j:
                    xj_sum += K_array[row][col]

        #居中核矩阵的核函数
        KC_array[i][j] = k-xi_sum/150-xj_sum/150+ np.sum(K_array)/(150**2)




KC_array = np.array(KC_array)
print("居中核矩阵")
print(KC_array)
print("--------------------------------------------------")


# 初始化规范核矩阵
KZ_array = []
for i in range(0,row_number):
    list = []
    for j in range(0,row_number):
        list.append(0)
    KZ_array.append(list)


for i in range(0,row_number):
    for j in range(0,row_number):

        k =  KC_array[i][j]
        k1 = KC_array[i][i]
        k2 = KC_array[j][j]

        # 矩阵规范化核函数
        KZ_array[i][j] = round(k / sqrt(k1 * k2), 6)


KZ_array = np.array(KZ_array)
print("特征空间中居中和归一化点的成对点积")
print(KZ_array)
print("--------------------------------------------------")

'''''''''
# 求协方差矩阵
covariance_matrix = np.cov(KZ_array)

print("协方差矩阵")
print(covariance_matrix)

eig_val, eig_vec = np.linalg.eig(covariance_matrix)
print('特征值：\n{}'.format(eig_val[0:2]))
print('特征向量：\n{}'.format(eig_vec[0:2]))

print("----------------------------------------------------")
# 求特征值和特征向量

new_data = np.transpose(np.dot(eig_vec[0:2],np.transpose(covariance_matrix)))

print(new_data)

plt.scatter(new_data[:,0],new_data[:,1])

plt.show()

'''''''''