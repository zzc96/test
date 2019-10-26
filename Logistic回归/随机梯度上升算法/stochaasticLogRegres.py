#-*- coding:utf-8 -*-
import math
import numpy
import matplotlib.pyplot as plt
import random

weightsHis = []

#导入数据
def loadDataSet():
    dataMat = []                                                     #数据
    labelMat = []                                                    #标签

    fr = open('./testSet.txt')                                       #打开文件
    for line in fr.readlines():                                      #读取每一行
        lineArr = line.strip().split()                               #移除字符串头尾的空格和换行符，然后进行分割
        #其中X0默认为1，实为初始化biase，将直线方程设为y=w*x+b
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])    #设置数据集分别对应X1 X2 其中为了方便计算，将X0设置为1.0
        labelMat.append(int(lineArr[2]))                             #设置对应的类别标签

    return dataMat,labelMat

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
            weightsHis.append(weights)                                      #为了画出曲线查看收敛的情况，将数据先保存
            del(dataIndex[randIndex])

    return weights

def plotBestFit(wei):
    weights = wei                                                   #得到权重
    dataMat,labelMat = loadDataSet()                                #读取原始数据和标签
    dataArr = numpy.array(dataMat)                                  #转换成数组
    n = numpy.shape(dataArr)[0]                                     #得到数据长度
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):                                             #按类别找出数据点，为绘画做准备
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])                           #
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()                                             #创建图形实例
    ax = fig.add_subplot(111)                                      #初始化画布 1行1列，第1块
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')              #绘制散点图，红色
    ax.scatter(xcord2,ycord2,s=30,c='green')                       #绘制散点图，绿色
    x = numpy.arange(-3.0,3.0,0.1)                                 #x从-3到3，步长是0.1
    y = (-weights[0] - weights[1] * x)/weights[2]                  #0=w0x0+w1x1+w2x2 (x0=1) 此处x=x1；y=x2
    ax.plot(x,y)                                                   #以x纵坐标，计算y，并且绘制y
    plt.xlabel('X1')                                               #X轴标签
    plt.ylabel('X2')                                               #Y轴标签
    plt.show()                                                     #显示


def plotWaveform():
    fig = plt.figure()  # 创建图形实例

    m, n = numpy.shape(weightsHis)
    x = numpy.array(range(m))

    ax0 = fig.add_subplot(311)  # 初始化画布 3行1列，第1块
    plt.xlabel("times")
    plt.ylabel("X0")
    y0 = []
    for i in range(m):
        y0.append(weightsHis[i][0])
    ax0.plot(x,y0)


    ax1 = fig.add_subplot(312)  # 初始化画布 3行1列，第2块
    plt.xlabel("times")
    plt.ylabel("X1")
    y1 = []
    for i in range(m):
        y1.append(weightsHis[i][1])
    ax1.plot(x, y1)

    ax2 = fig.add_subplot(313)  # 初始化画布 3行1列，第3块
    plt.xlabel("times")
    plt.ylabel("X2")
    y2 = []
    for i in range(m):
        y2.append(weightsHis[i][2])
    ax2.plot(x, y2)

    plt.show()

if __name__ == "__main__":
    dataArr,labelMat = loadDataSet()
    weights = stocGradAscent(numpy.array(dataArr),labelMat,numIter=200)
    plotBestFit(weights)
    plotWaveform()