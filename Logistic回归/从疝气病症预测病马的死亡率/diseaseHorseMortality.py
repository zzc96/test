#-*- coding:utf-8 -*-
import math
import numpy
import matplotlib.pyplot as plt
import random

#如果特征缺失，那么可以用下列可选的做法
'''
1.使用可用特征的均值来填补缺失值
2.使用特殊值来填补缺失值，如-1
3.忽略有缺失值的样本
4.使用相似样本的均值添补缺失值
5.使用另外的机器学习算法预测缺失值
'''

#定义sigmoid公式 1/（1 + e(-z)）
def sigmoid(inX):
    return 1.0/(1 + numpy.exp(-inX))

#随机梯度上升算法
#dataMatIn  数据集  每列代表不同的特征，每行代表每个训练样本
#classLabels  标签集
def stocGradAscent(dataMatrix,classLabels,numIter = 150):
    m,n = numpy.shape(dataMatrix)                                   #得到矩阵的大小
    weights = numpy.ones(n)                                         #初始化W为1
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            #alpha在此处的做法是为了缓解数据的波动或者高频波动，
            #存在常数项是为了保证在多次迭代智慧新数据仍然具有一定的影响
            alpha = 4/(1.0 + j + i) + 0.038                          #alpha随着迭代次数不断减少，但不会小到0
            #随机选取样本来更新回归系数，可减少参数的周期性波动
            randIndex = int(random.uniform(0,len(dataIndex)))               #在数据集中随机选取一个样本   randIndex为下标
            h = sigmoid(sum(dataMatrix[randIndex] * weights))               #计算预测类别
            error = classLabels[randIndex] - h                              #计算真实类别与预测类别的差值
            weights = weights + alpha * dataMatrix[randIndex] * error       #按真实类别和预测类别的差值的方向调整回归系数
            del(dataIndex[randIndex])

    return weights

#分类输出
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))                #weights是训练好的权重，inX是要测试的数据   使用sigmoid对要测试的数据和训练好的权重进行计算
                                                    #得出测试结果
    if prob > 0.5:                                  #如果结果大于0.5   则认为是1类
        return 1.0
    else:                                           #否则认为是0类
        return 0.0


# def modelTrain():
#     frTrain = open('./horseColicTraining.txt')
#
#     trainingSet = []
#     trainingLabels = []
#     for line in frTrain.readlines():
#         currLine = line.strip().split('\t')
#         lineArr = []
#         for i in range(21):
#             lineArr.append(float(currLine[i]))
#         trainingSet.append(lineArr)
#         trainingLabels.append(float(currLine[21]))
#
#     trainWeights = stocGradAscent(numpy.array(trainingSet), trainingLabels, 500)
#     return trainWeights

def colicTest():
    frTest = open("./horseColicTest.txt")                              #打开测试集数据
    frTrain = open('./horseColicTraining.txt')                         #打开训练集数据

    trainingSet = []                                                   #训练集数据列表
    trainingLabels = []                                                #训练集标签列表
    for line in frTrain.readlines():                                   #读取每一行数据
        currLine = line.strip().split('\t')                            #去掉头尾空格，并且以'\t'作为分割
        lineArr = []                                                   #每一行特征暂存
        for i in range(21):                                            #一共21个特征
            lineArr.append(float(currLine[i]))                         #每一行每一个特征加入行特征
        trainingSet.append(lineArr)                                    #写入训练集数据列表
        trainingLabels.append(float(currLine[21]))                     #读取标签

    trainWeights = stocGradAscent(numpy.array(trainingSet), trainingLabels, 500)   #进行500次迭代训练
    errorCount = 0                                                     #定义错误率
    numTestVec = 0.0                                                   #定义测试两
    for line in frTest.readlines():                                    #读取测试数据的每一行
        numTestVec += 1.0                                              #每一行，测试量+1
        currLine = line.strip().split('\t')                            #去掉头尾空格，并且以‘\t’作为分割
        lineArr = []                                                   #行特征定义
        for i in range(21):                                            #轮询21个行特征
            lineArr.append(float(currLine[i]))                         #每个特征加入行特征
        if int(classifyVector(numpy.array(lineArr),trainWeights)) != int(currLine[21]):  #进行测试，如果不相等，则测试错误
            errorCount += 1                                            #错误次数+1
        errorRate = (float(errorCount)/numTestVec)                     #计算错误率
        print("the error rate of this test is:%f" % errorRate)
        return errorRate                                               #返回错误率

#测试函数
def multiTest():
    numTest = 20                                                #测试次数
    errorSum = 0.0                                              #错误率
    for k in range(numTest):                                    #轮询测试
        errorSum += colicTest()                                 #计算总错误率

    print("after %d iterations the average error rate is:%f" % (numTest,errorSum/float(numTest)))  #计算平均错误率


if __name__ == "__main__":
    # trainWeights = modelTrain()
    # print(trainWeights)
    multiTest()