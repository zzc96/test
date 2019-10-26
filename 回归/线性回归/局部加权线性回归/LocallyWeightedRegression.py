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
def plotPoint(xArr,yArr,yHat):
    xMat = numpy.mat(xArr)                                      #将数据集转换成矩阵
    yMat = numpy.mat(yArr)                                      #将标签集转换成矩阵

    srtInd = xMat[:,1].argsort(0)                               #去除第一列，并且进行排序
    xSort = xMat[srtInd][:,0,:]

    fig = plt.figure()                                          #获得画图对象实例
    ax = fig.add_subplot(111)                                   #创建一行一列的画布，并且当前位置为1
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T.flatten().A[0],s=2,c="red")
    plt.show()

#计算相关系数，判定模型的好坏，相关系数越大，模型越好
#计算两个序列的相关系数就是计算预测值yHat序列和真实值y序列的匹配程度
def calcCorrcoef(xArr,yArr,ws):
    xMat = numpy.mat(xArr)                                       #将数据集转换成矩阵
    yMat = numpy.mat(yArr)                                       #将标签集转换成矩阵
    yHat = xMat * ws                                             #y = ws[0] + ws[1]*X1   假定X0为1

    corrcoef = numpy.corrcoef(yHat.T,yMat)                       #计算相关系数，真实值和预测值的相关系数
    print("corrcoef:",corrcoef)


def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = numpy.mat(xArr)                                      #将数据集转换成矩阵
    yMat = numpy.mat(yArr).T                                    #将数据转换成矩阵，并且进行转置
    m = numpy.shape(xMat)[0]                                    #获得数据集的数量
    #权重矩阵是一个方阵，阶数等于样本点个数，也就是说，该矩阵为每个样本点初始化了一个权重
    weights = numpy.mat(numpy.eye((m)))                         #创建对角权重矩阵weights
    #遍历数据集，计算每个样本点对应的权重值，随着样本点于待测点距离的递增，权重将以指数级衰减
    for j in range(m):
        #计算样本点和待预测点的距离
        diffMat = testPoint - xMat[j,:]
        #w(i,i) = exp(|Xi - X| / (-2k**2))
        weights[j,j] = numpy.exp(diffMat * diffMat.T/(-2.0 * k**2))  #参数k控制衰减的速度
    #计算w = (XtWX)**-1XtWy
    xTx = xMat.T * (weights * xMat)                             #计算XtWX
    if numpy.linalg.det(xTx) == 0.0:                            #判断行列式是否为0    numpy中linalg.det()计算行列式
        print("This matrix is singular,cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))                    #计算w
    print(ws)
    # 计算相关系数
    # calcCorrcoef(xArr, yArr, ws)
    return testPoint * ws                                       #返回估计

#lwlr测试函数
def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = numpy.shape(testArr)[0]                                 #获得测试集数目
    yHat = numpy.zeros(m)                                       #初始化预测集
    for i in range(m):                                          #轮询数据
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

if __name__ == "__main__":
    xArr,yArr = loadDataSet("./ex0.txt")
    # yHat = lwlrTest(xArr, xArr, yArr, 1)
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    # yHat = lwlrTest(xArr,xArr,yArr,0.003)
    print(yHat)
    plotPoint(xArr,yArr,yHat)