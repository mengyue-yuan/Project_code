import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


# 加载数据
data = np.loadtxt("magic04.txt", delimiter=",", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

label_list =["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist"]

#列数
col_number = np.size(data, axis=1)
#行数
row_number = np.size(data, axis=0)

# 1
# 计算多元均值向量
mean_vector = np.mean(data, axis=0).reshape(col_number, 1)
print("Mean Vector均值向量\n", mean_vector.tolist(), "\n")

t_mean_vector = np.transpose(mean_vector)

# 2
# 计算中心数据矩阵
centered_data_matrix = data - (1 * t_mean_vector)
# 中心数据矩阵的转置
centered_matrix_tr = np.transpose(centered_data_matrix)
# 计算中心数据矩阵各列之间的内积
covariance_matrix_inner = (1 / row_number) * np.dot(centered_matrix_tr, centered_data_matrix)

print(
    "The sample covariance matrix as inner products between the columns of the centered data matrix \n",
    "将样本协方差矩阵计算为中心数据矩阵的各列之间的内积",
    covariance_matrix_inner, "\n")

# 3
# 计算样本协方差矩阵，作为中心数据点之间的外积
sum = np.zeros(shape=(col_number, col_number))
for i in range(0, row_number):
    sum += np.dot(np.reshape(centered_matrix_tr[:, i], (-1, 1)),
                np.reshape(centered_data_matrix[i, :], (-1, col_number)))

covariance_matrix_outer = (1 / row_number) * sum

print(
    "The sample covariance matrix as outer product between the centered data points \n",
    "计算样本协方差矩阵作为中心数据点之间的外部乘积 \n",
    covariance_matrix_outer, "\n")

# 4
# 计算属性1和2之间的相关性
# 属性1
vector1 = np.array(centered_data_matrix[:, 0])
# 属性2
vector2 = np.array(centered_data_matrix[:, 1])

# 计算属性向量之间的角度
def angle(vec1, vec2):
    # 转化为单位向量
    v1 = vec1 / np.linalg.norm(vec1)
    v2 = vec2 / np.linalg.norm(vec2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

# 计算两个属性之间的夹角余弦值
correlation = math.cos(angle(vector1, vector2))
print(" 属性1、2夹角余弦值: %.5f" % correlation, "\n")


# 6
# 哪个属性的方差最大，哪个属性的方差最小
# 计算方差
variance_vector = np.var(data, axis=0)
max_var = np.max(variance_vector)
min_var = np.min(variance_vector)
# 方差最大的属性
for i in range(0, col_number):
    if variance_vector[i] == max_var:
        max_var_index = i
# 方差最小的属性
for i in range(0, col_number):
    if variance_vector[i] == min_var:
        min_var_index = i

print(" 属性 %s 方差最大 = %.3f " % (label_list[max_var_index],max_var))
print(" 属性 %s 方差最小 = %.3f " % (label_list[min_var_index],min_var))


# 7
# 哪对属性的协方差最大，哪对属性的协方差最小
# 计算协方差矩阵
covariance_matrix = np.cov(data, rowvar=False)
max_cov=np.max(covariance_matrix)
min_cov=np.min(covariance_matrix)

# 找到协方差最大的一对属性
for i in range(0, col_number):
    for j in range(0, col_number):
        if covariance_matrix[i, j] == max_cov:
            max_cov_atrr1=i
            max_cov_attr2=j
# 找到协方差最小的一对属性
for i in range(0, col_number):
    for j in range(0, col_number):
        if covariance_matrix[i, j] == min_cov:
            min_cov_atrr1 = i
            min_cov_attr2 = j


print("最大协方差 = %.3f 为属性 %s 与 %s" %(max_cov,label_list[max_cov_atrr1],label_list[max_cov_attr2]))      # finding index of max covariance
print("最小协方差 = %.3f 为属性 %s 与 %s" %(min_cov,label_list[min_cov_atrr1],label_list[min_cov_attr2]))      # finding index of min covariance


# 5
# 假设属性1是正态分布的，则绘制其概率密度函数。
# 选取属性1
df = pd.DataFrame(data[:, 0])
# 绘制概率密度曲线
df.plot(kind='density',xlim=[-50,160],legend = None)
plt.show()


