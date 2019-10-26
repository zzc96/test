import numpy
import matplotlib.pyplot as plt

#打开一个用tab键分割的文本文件
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1    #得到每一行有多少个特征，最后一个值默认是目标值
    dataMat = []                                                #初始化数据集为空
    labelMat = []                                               #初始化标签集为空
    fr = open(fileName)                                         #打开文件

    for line in fr.readlines():                                 #读取每一行
        lineArr = []                                            #特征中间量
        curLine = line.strip().split('\t')                      #分割每一行
        for i in range(numFeat):                                #按每一行特征数寻找特征
            lineArr.append(float(curLine[i]))                   #找出特征
        dataMat.append(lineArr)                                 #添加到特征集
        labelMat.append(float(curLine[-1]))                     #添加到标签集

    return dataMat,labelMat                                     #返回数据集和标签集

#画出点和拟合曲线
def plotPoint(ridgeWeights):
    fig = plt.figure()                                          #获得画图对象实例
    ax = fig.add_subplot(111)                                   #创建一行一列的画布，并且当前位置为1
    ax.plot(ridgeWeights)
    plt.xlabel("log(lambda)")
    plt.xlim(0,30)
    plt.ylim(-1.0,2.5)
    plt.show()

#计算相关系数，判定模型的好坏，相关系数越大，模型越好
#计算两个序列的相关系数就是计算预测值yHat序列和真实值y序列的匹配程度
def calcCorrcoef(xArr,yArr,ws):
    xMat = numpy.mat(xArr)                                       #将数据集转换成矩阵
    yMat = numpy.mat(yArr)                                       #将标签集转换成矩阵
    yHat = xMat * ws                                             #y = ws[0] + ws[1]*X1   假定X0为1

    corrcoef = numpy.corrcoef(yHat.T,yMat)                       #计算相关系数，真实值和预测值的相关系数
    print("corrcoef:",corrcoef)


#标准化函数，将数据集进行标准化
def regularize(xMat):
    inMat = xMat.copy()                                         #将数据进行一份拷贝
    inMeans = numpy.mean(inMat,0)                               #mean()求取均值，axis=0：压缩行，对各列求均值，返回1*n矩阵
    inVar = numpy.var(inMat,0)                                  #求数据集各列方差的无偏估计值（N-1）
    inMat = (inMat - inMeans)/inVar                             #特征减去均值除以方差，进行数据标准化
    return inMat                                                #返回标准化的数据集


#分析预测误差的大小，误差总和
def rssError(yArr,yHatArr):
    return ((yArr - yHatArr)**2).sum()


#逐步线性回归算法的实现
#eps  表示每次迭代需要调整的步长
#numIt 表示迭代次数
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    #数据标准化  对特征做标准化处理，使每维特征具有相同的重要性
    xMat = numpy.mat(xArr)                                      #将数据集转换成矩阵
    yMat = numpy.mat(yArr).T                                    #将标签集转换成矩阵并且进行转置
    yMean = numpy.mean(yMat,0)                                  #计算特征均值
    yMat = yMat - yMean                                         #计算新的标签
    # 所有特征都减去各自的均值并除以方差
    #把特征按照均值为0方差为1进行标准化处理
    xMat = regularize(xMat)                                     #对数据集进行标准化处理
    m,n = numpy.shape(xMat)                                     #获得数据集行数和特征数
    returnMat = numpy.zeros((numIt,n))
    #创建一个向量ws保存w的值
    ws = numpy.zeros((n,1))                                     #初始化全部权重为0
    wsTest = ws.copy()
    wsMax = ws.copy()
    #再每轮迭代过程中，迭代numIt次
    for i in range(numIt):
        # print(ws.T)
        lowestError = numpy.inf                                 #设置当前最小误差lowestError为正无穷
        #贪心算法在所有特征上运行两次for循环，分别计算增加或减少该特征对误差的影响
        for j in range(n):                                      #轮询每一个特征
            for sign in [-1,1]:                                 #用于增大或者缩小
                wsTest = ws.copy()                              #特征拷贝
                wsTest[j] += eps * sign                         #增大或者缩小 ，获得新的W
                yTest = xMat * wsTest                           #计算新W下的预测值
                rssE = rssError(yMat.A,yTest.A)                 #计算误差，这里使用的是平方误差
                #初始误差值设为正无穷，经过与所有的误差比较之后取最小的误差
                if rssE < lowestError:                          #如果误差小于当前最小误差lowestError，
                    lowestError = rssE                          #更新当前最小误差
                    wsMax = wsTest                              #记录回归系数
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat


if __name__ == "__main__":
    xArr,yArr = loadDataSet("abalone.txt")
    ridgeWeights = stageWise(xArr,yArr,0.005,1000)
    print(ridgeWeights)
    plotPoint(ridgeWeights)
