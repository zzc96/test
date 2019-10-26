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


#用于计算回归系数
#实现在给定lam下的岭回归求解，  w = (Xt*X+rI)(-1) *Xt * y
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T * xMat                                         #计算Xt*X
    denom = xTx + numpy.eye(numpy.shape(xMat)[1]) * lam         #计算Xt*X+rI   numpy中eye生成单位矩阵
    #如果lam设置为0的时候，一样会产生错误，所以需要检查行列式是否为零
    if numpy.linalg.det(denom) == 0.0:                          #判断行列式是否为0    numpy中linalg.det()计算行列式
        print("This matrix is singular,cannot do inverse")
        return
    #如果矩阵非奇异就计算回归系数并返回
    ws = denom.I * (xMat.T * yMat)                              #计算w = (Xt*X+rI)(-1) *Xt * y
    return ws                                                   #返回ws

#用于在一组r上测试结果
def ridgeTest(xArr,yArr):
    #对特征做标准化处理，使每维特征具有相同的重要性
    xMat = numpy.mat(xArr)                                      #将数据集转换成矩阵
    yMat = numpy.mat(yArr).T                                    #将标签集转换成矩阵，并且进行转置
    yMean = numpy.mean(yMat,0)                                  #计算特征均值
    yMat = yMat - yMean                                         #计算新的标签
    #所有特征都减去各自的均值并除以方差
    xMeans = numpy.mean(xMat,0)                                 #mean()求取均值，axis=0：压缩行，对各列求均值，返回1*n矩阵
    xVar = numpy.var(xMat,0)                                    #求数据集各列方差的无偏估计值（N-1）
    xMat = (xMat - xMeans)/xVar                                 #特征减去均值除以方差，进行数据标准化
    #在30个不同的lam下调用ridgeRegres()函数，这里的lam以指数级变化，可以看出lam在去非常小和非常大的值时对结果造成的影响
    numTestPts = 30
    wMat = numpy.zeros((numTestPts,numpy.shape(xMat)[1]))       #创建初始化回归系数矩阵
    for i in range(numTestPts):                                 #在30个lam下测试
        ws = ridgeRegres(xMat,yMat,numpy.exp(i-10))             #lam以指数级变化
        print(numpy.exp(i-10))
        wMat[i,:] = ws.T                                        #添加到回归系数矩阵
    return wMat                                                 #返回


#分析预测误差的大小，误差总和
def rssError(yArr,yHatArr):
    return ((yArr - yHatArr)**2).sum()

if __name__ == "__main__":
    abX,abY = loadDataSet("abalone.txt")
    ridgeWeights = ridgeTest(abX,abY)
    print(ridgeWeights)
    plotPoint(ridgeWeights)
