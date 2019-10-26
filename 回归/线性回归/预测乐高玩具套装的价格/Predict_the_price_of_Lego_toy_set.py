import numpy
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup


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

#从html文件中提取相关信息
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    # 打开并读取HTML文件
    fr = open(inFile,encoding='utf-8')
    soup = BeautifulSoup(fr.read())
    i = 1

    # 根据HTML页面结构进行解析
    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow)!=0):
        currentRow = soup.findAll('table', r="%d" % i)
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()

        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0

        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde)==0:
            print("item #%d did not sell" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','')
            priceStr = priceStr.replace(',','')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)

            # 去掉不完整的套装价格
            if  sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)


#读取本地的网页，获取信息
#outFile  是提取出来的信息存放的新文件
#依次读取六种乐高套装的数据，并生成数据矩阵
def setDataCollect(retX, retY):
    scrapePage(retX, retY, 'setHtml\lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, 'setHtml\lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, 'setHtml\lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, 'setHtml\lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, 'setHtml\lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, 'setHtml\lego10196.html', 2009, 3263, 249.99)


#交叉验证和岭回归测试
#lgX和lgY存有数据集中的X和Y值的list对象，默认值为10
#numVal是算法中交叉验证的次数，默认值为10
def crossValidation(xArr,yArr,numVal=10):
    m = len(yArr)                                                    #获取数据集的长度
    indexList = list(range(m))                                       #获得索引
    errorMat = numpy.zeros((numVal,30))                              #创建错误矩阵，轮询计算30次回归，30个回归系数
    for i in range(numVal):
        #创建训练集和测试集容器
        trainX = []
        trainY = []
        testX = []
        testY = []
        #使用numpy中的random.shuffle()对数据中的元素进行混洗（打乱原来的顺序）
        #可以实现训练集和测试集数据点的随机选取
        numpy.random.shuffle(indexList)
        #轮询所有数据集
        for j in range(m):
            #将数据集的90%用于训练集
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])                    #训练集数据
                trainY.append(yArr[indexList[j]])                    #训练集标签
            #其余的10%用于测试集
            else:
                testX.append(xArr[indexList[j]])                     #测试集
                testY.append(yArr[indexList[j]])                     #测试集标签
        #建立新的矩阵来保存岭回归中的所有回归系数，使用30组回归系数来循环测试回归效果
        wMat = ridgeTest(trainX,trainY)
        #30次回归系数计算回归结果
        for k in range(30):
            #测试数据需要使用与训练集相同的参数来执行标准化
            matTestX = numpy.mat(testX)                              #将测试数据转换成矩阵
            matTrainX = numpy.mat(trainX)                            #将训练集转换成矩阵
            meanTrain = numpy.mean(matTrainX,0)                      #求测试数据的均值
            varTrain = numpy.var(matTrainX,0)                        #求数据集各列方差的无偏估计值（N-1）
            matTestX = (matTestX - meanTrain)/varTrain               #特征减去均值除以方差，进行测试数据标准化
            yEst = matTestX * numpy.mat(wMat[k,:]).T + numpy.mean(trainY)    #计算预测值
            errorMat[i,k] = rssError(yEst.T.A,numpy.array(testY))    #计算误差，并且将误差保存在errorMat中

    #errorMat保存了每个lam对应的多个误差值，
    #计算岭回归的最佳模型
    meanErrors = numpy.mean(errorMat,0)                              #计算误差估计值的均值
    minMean = numpy.float(min(meanErrors))                           #找出均值最小的
    bestWeights = wMat[numpy.nonzero(meanErrors == minMean)]         #返回相等的权重，寻找最好的权重值
    xMat = numpy.mat(xArr)                                           #将数据集转换成矩阵
    yMat = numpy.mat(yArr).T                                         #将标签转换成矩阵，并且进行转置
    meanX = numpy.mean(xMat,0)                                       #计算数据集的均值
    varX = numpy.var(xMat,0)                                         #求数据集各列方差的无偏估计值（N-1）
    unReg = bestWeights/varX                                         #计算岭回归的最佳模型
    print("the best model from Ridge Regression is:\n",unReg)
    print("with constant term:",-1 * numpy.sum(numpy.multiply(meanX,unReg)) + numpy.mean(yMat))


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


if __name__ == "__main__":
    lgX = []
    lgY = []
    setDataCollect(lgX,lgY)
    # print(lgX)
    # m,n = numpy.shape(lgX)
    # lgX1 = numpy.mat(numpy.ones((m,5)))
    # lgX1[:,1:5] = numpy.mat(lgX)
    # print(lgX1[0])
    crossValidation(lgX,lgY,10)
    ws = ridgeTest(lgX,lgY)
    print(ws)


