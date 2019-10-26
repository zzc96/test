#-*- coding:utf-8 -*-
import math
import numpy
import matplotlib.pyplot as plt

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

#梯度上升算法
#dataMatIn  数据集  每列代表不同的特征，每行代表每个训练样本
#classLabels  标签集
def gradAscent(dataMatIn,classLabels):
    dataMatrix = numpy.mat(dataMatIn)                               #将数据集转换成Numpy矩阵
    labelMat = numpy.mat(classLabels).transpose()                   #将标签转换成矩阵并且将原向量转置
    m,n = numpy.shape(dataMatrix)                                   #得到矩阵的大小
    alpha = 0.001                                                   #初始化步长为0.001
    maxCycles = 500                                                 #迭代次数为500次
    weights = numpy.ones((n,1))                                     #初始化W为1
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)                           #计算预测类别
        error = (labelMat - h)                                      #计算真实类别与预测类别的差值
        weights = weights + alpha * dataMatrix.transpose() * error  #按真实类别和预测类别的差值的方向调整回归系数

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

if __name__ == "__main__":
    dataArr,labelMat = loadDataSet()
    weights = gradAscent(dataArr,labelMat)
    #x为array格式，weights为matrix格式，使用getA()方法，将matrix格式转换成array格式
    plotBestFit(weights.getA())