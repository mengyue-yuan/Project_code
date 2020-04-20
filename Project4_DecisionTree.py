from numpy import *
from Project4_best_split_point import get_best_split_point

data = loadtxt("/Users/yuanmengyue/Desktop/2020春季课程/万林/datasets/iris.txt",delimiter = ',',usecols = (0,1,2,3),dtype = float)
label = loadtxt("/Users/yuanmengyue/Desktop/2020春季课程/万林/datasets/iris.txt",delimiter = ',',usecols = (4),dtype = str)



def Decision_tree(grade,data,label,min_point_number=5,purity=0.95):

    #记录树的等级
    grade = grade

    # 统计现有集合中点的数量
    leafs_size = len(data)
    # 创建一个字典{种类：数量}
    dic = {}
    for curlabel in label:
        if curlabel not in dic.keys():
            dic[curlabel] = 1
        else:
            dic[curlabel] += 1
    # 返回出现次数最多的属性
    main_category = max(dic, key=lambda k: dic[k])
    # 计算空间纯度：出现概率最大的类别的概率
    pxi = float(dic[main_category])/len(label)

    # 终止条件
    if leafs_size <= min_point_number or pxi >=purity:
        # 将这个分区内的点都归为概率最大类
        # tap用于打印不同等级决策的缩进
        tap = ''
        add_tap = str('   ')
        for i in range(0, grade):
            tap += add_tap
        # 输出叶子节点
        print(tap,"叶：label：",main_category,' purity=',pxi,' size=',leafs_size)
        # 停止递归
        return 0


    feature_type = ['花萼长度','花萼宽度','花瓣长度','花瓣宽度']

    #初始化
    best_feature_index = 0
    best_spilt_point = 0
    max_gain = 0


    # 遍历所有属性
    for feature_number in range(0,len(feature_type)):
        feature_list = []

        for i in range(0, len(data)):
            feature_list.append(data[i][feature_number])
        # 计算此类别下最佳分割点和信息增益
        spilt_point, gain = get_best_split_point(feature_list, label)

        if gain > max_gain:
            max_gain = gain
            best_feature_index = feature_number
            best_spilt_point = spilt_point

    # 根据树的等级输出
    tap = ''
    add_tap = str('   ')
    for i in range(0,grade):
        tap  += add_tap

    print(tap,"决策：",feature_type[best_feature_index]," <= ",best_spilt_point,",增益= ",max_gain)


    # 根据最佳分类属性和最佳分割点对数据进行划分
    DY_data_list = []  #小于分割点的数据集合
    DN_data_list = []  #大于等于分割点的数据集合
    DY_label_list = []  #小于分割点的数据标签集合
    DN_label_list = []  #大于等于分割点的数据标签集合

    for i in range(0,len(data)):
        if data[i][best_feature_index]<=best_spilt_point:
            DY_data_list.append(data[i])
            DY_label_list.append(label[i])
        else:
            DN_data_list.append(data[i])
            DN_label_list.append(label[i])

    grade +=1 # 用于记录决策的等级，多一次递归则+1
    # 对DY集合继续分割
    Decision_tree(grade,DY_data_list, DY_label_list, min_point_number=5, purity=0.95)
    # 对DN集合继续分割
    Decision_tree(grade,DN_data_list, DN_label_list, min_point_number=5, purity=0.95)



if __name__ == '__main__':
    grade = 0
    Decision_tree(grade, data, label, min_point_number=5, purity=0.95)




