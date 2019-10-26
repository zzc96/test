from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt


# inX 用于分类的输入向量
# dataSet 输入的训练样本集
# labels 标签向量
# k 选择最近邻的数目
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 矩阵相减
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 差值平方
    sqDiffMat = diffMat ** 2
    # 平方和 axis=1表示将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开平方
    distances = sqDistances ** 0.5
    # 排序
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 找出前k个值
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 统计前k次标签出现的次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回最大次数的标签
    return sortedClassCount[0][0]


#读取并且进行数据处理操作
def file2matrix(filename):
    fr = open(filename)                                           #打开文件
    numberOfLines = len(fr.readlines()) - 1                       #获得文件的行数  数据集的数量  -1是应为第一行是标题
    returnMat = zeros((numberOfLines, 30))                        #创建以零填充的特征矩阵，特征数量为30个，数据集为numberOfLines行
    classLabelVector = []                                         #创建初始化标签列表
    fr = open(filename)                                           #打开文件
    index = 0                                                     #索引
    lineFlag = 0                                                  #是否是第一行的标记，应为第一行是标题，我们不取
    for line in fr.readlines():                                   #轮询读取文件
        if lineFlag == 0:                                         #如果是第一行
            lineFlag = 1                                          #去掉标记，开始读取
            continue
        line = line.strip()                                       #截取所有的回车字符
        listFromLine = line.split('\t')                           #使用tab字符\t将行数据分割成一个元素列表
        returnMat[index, :] = listFromLine[2:]                    #每一行特征的前两个分别是ID和标签，不取
        if listFromLine[1] == "B":                                #读取标签，如果是B
            classLabelVector.append(1)                            #标记为1
        elif listFromLine[1] == "M":                              #如果是M
            classLabelVector.append(2)                            #标记为2
        index += 1                                                #行数加1
    return returnMat, classLabelVector                            #返回特征和标签


def mat2Show(datingDataMat, datingLabels):
    # datingDataMat,datingLabels = file2matrix(filename)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    plt.xlabel("Frequent flyer miles earned each year")
    plt.ylabel("% of time spent playing video games")
    # plt.ylim(0,2)
    plt.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # 利用变量datingLabels存储的类标签属性，在散点图上绘制色彩不等、尺寸不同的点
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
    plt.show()


# 函数用于将数值归一化，将取值范围处理为0-1间
# 借助newValue = (oldValue-min)/(max-min)
def autoNorm(dataSet):
    #将每列最小值放在变量中    0表示从列中选择最小值
    minVals = dataSet.min(0)
    #将每列最大值放在变量中    0表示从列中选择最大值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))               #初始化数据集大小的全0的数组，用以存放处理后的数据集
    print(normDataSet)
    # 获取dataSet的行数
    m = dataSet.shape[0]
    print(m)
    # tile(minVals, (m,1))表明在列方向上重复1次，在行方向上重复m次
    normDataSet = dataSet - tile(minVals, (m, 1))  # oldValue-min
    normDataSet = normDataSet / tile(ranges, (m, 1))  # (oldValue-min)/(max-min)
    print(normDataSet)
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10  # 使用10%数据做测试用
    datingDataMat, datingLabels = file2matrix('wdbc.data.txt')  # 读取数据并且进行格式转换
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 数据归一化
    mat2Show(datingDataMat, datingLabels)  # 数据散点图显示
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)                              #获得测试数据两
    errorCount = 0.0
    # 评估错误率
    for i in range(numTestVecs):
        # 测试10%的量
        #取测试集中的第i个，放入训练集中进行测试，测试集为0—numTestVecs   训练集为numTestVecs-m（总数），取前3个数据
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        #如果测试输出标签不等于真实标签，表明测试出错，记录
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    #错误数/测试总数=测试错误率
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


def datingClass(predicData):
    resultList = ["B", "M"]
    datingDataMat, datingLabels = file2matrix('wdbc.data.txt')  # 读取数据并且进行格式转换
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 数据归一化
    classifierResult = classify0(((predicData - minVals) / ranges), normMat, datingLabels, 3)  # k近邻算法预测

    print("result:%s" % (resultList[classifierResult - 1]))


if __name__ == "__main__":
    datingClassTest()  # 算法评估
    datingClass([12.47,17.31,80.45,480.1,0.08928,0.0763,0.03609,0.02369,\
                 0.1526,0.06046,0.1532,0.781,1.253,11.91,0.003796,0.01371, \
                 0.01346,0.007096,0.01536,0.001541,14.06,24.34,92.82,607.3, \
                 0.1276,0.2506,0.2028,0.1053,0.3035,0.07661])  # 算法使用