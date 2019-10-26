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

#用来计算最佳拟合直线
def standRegres(xArr,yArr):
    xMat = numpy.mat(xArr)                                      #将数据转换成矩阵
    yMat = numpy.mat(yArr).T                                    #将数据转换成矩阵，并且进行转置
    xTx = xMat.T * xMat                                         #计算Xt*X
    #如果没有检查行列式是否为零就试图计算矩阵的逆，就会出现错误
    if numpy.linalg.det(xTx) == 0.0:                            #判断行列式是否为0    numpy中linalg.det()计算行列式
        print("This matrix is singular,cannot do inverse")      #如果行列式为0，则返回
        return
    ws = xTx.I * (xMat.T * yMat)                                #计算并返回w
    #使用numpy的线性代数库的函数求解未知矩阵
    # ws = numpy.linalg.solve(xTx,xMat.T * yMat.T)
    return ws

#画出点和拟合曲线
def plotPoint(xArr,yArr,ws):
    xMat = numpy.mat(xArr)                                      #将数据集转换成矩阵
    yMat = numpy.mat(yArr)                                      #将标签集转换成矩阵
    yHat = xMat * ws                                            #y = ws[0] + ws[1]*X1   假定X0为1

    fig = plt.figure()                                          #获得画图对象实例
    ax = fig.add_subplot(111)                                   #创建一行一列的画布，并且当前位置为1
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()

#计算相关系数，判定模型的好坏，相关系数越大，模型越好
#计算两个序列的相关系数就是计算预测值yHat序列和真实值y序列的匹配程度
def calcCorrcoef(xArr,yArr,ws):
    xMat = numpy.mat(xArr)                                       #将数据集转换成矩阵
    yMat = numpy.mat(yArr)                                       #将标签集转换成矩阵
    yHat = xMat * ws                                             #y = ws[0] + ws[1]*X1   假定X0为1

    corrcoef = numpy.corrcoef(yHat.T,yMat)                       #计算相关系数，真实值和预测值的相关系数
    print("corrcoef:",corrcoef)


if __name__ == "__main__":
    xArr,yArr = loadDataSet("./ex0.txt")
    print(xArr[0:2])
    #ws存放的是回归系数
    ws = standRegres(xArr,yArr)
    print(ws)
    calcCorrcoef(xArr,yArr,ws)
    plotPoint(xArr,yArr,ws)