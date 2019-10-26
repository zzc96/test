from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt

#inX 用于分类的输入向量
#dataSet 输入的训练样本集
#labels 标签向量
#k 选择最近邻的数目
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #矩阵相减
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    #差值平方
    sqDiffMat = diffMat**2
    #平方和 axis=1表示将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    #开平方
    distances = sqDistances**0.5
    #排序
    sortedDistIndicies = distances.argsort()
    classCount={}
    #找出前k个值
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #统计前k次标签出现的次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回最大次数的标签
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #获得文件的行数
    returnMat = zeros((numberOfLines,3))        #创建以零填充的矩阵numpy，长度固定为3
    classLabelVector = []                       #创建标签列表
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()                     #截取所有的回车字符
        listFromLine = line.split('\t')         #使用tab字符\t将行数据分割成一个元素列表
        returnMat[index,:] = listFromLine[0:3]  #将前三个元素存储在特征矩阵中
        classLabelVector.append(int(listFromLine[-1]))  #将列表的最后一行（标签）存储到标签列表中
        index += 1
    return returnMat,classLabelVector

def mat2Show(datingDataMat,datingLabels):
    # datingDataMat,datingLabels = file2matrix(filename)
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    plt.xlabel("Frequent flyer miles earned each year")
    plt.ylabel("% of time spent playing video games")
    # plt.ylim(0,2)
    plt.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
    #利用变量datingLabels存储的类标签属性，在散点图上绘制色彩不等、尺寸不同的点
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
    plt.show()

#函数用于将数值归一化，将取值范围处理为0-1间
#借助newValue = (oldValue-min)/(max-min)
def autoNorm(dataSet):
    #将每列最小值放在变量中 0 表示从列中选择最小值
    minVals = dataSet.min(0)
    #将每列最大值放在变量中 0 表示从列中选择最大值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    print(normDataSet)
    #获取dataSet的行数
    m = dataSet.shape[0]
    print(m)
    #tile(minVals, (m,1)表明在列方向上重复1次，在行方向上重复m次
    normDataSet = dataSet - tile(minVals, (m,1))    #oldValue-min
    normDataSet = normDataSet/tile(ranges, (m,1))   #(oldValue-min)/(max-min)
    print(normDataSet)
    return normDataSet, ranges, minVals
   
def datingClassTest():
    hoRatio = 0.10      #使用10%数据做测试用
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #读取数据并且进行格式转换
    normMat, ranges, minVals = autoNorm(datingDataMat)                   #数据归一化
    mat2Show(datingDataMat,datingLabels)   #数据散点图显示
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    #评估错误率
    for i in range(numTestVecs):
        #测试10%的量
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)

def datingClass(predicData):
    resultList = ["not at all","in small doses","in large doses"]
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 读取数据并且进行格式转换
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 数据归一化
    classifierResult = classify0(((predicData - minVals)/ranges), normMat, datingLabels, 3)  #k近邻算法预测

    print("result:%s"%(resultList[classifierResult-1]))

if __name__ == "__main__":
    datingClassTest()  #算法评估
    datingClass([10000,2,1])  #算法使用