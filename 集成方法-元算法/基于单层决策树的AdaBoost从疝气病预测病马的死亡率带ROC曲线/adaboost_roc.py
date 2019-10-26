#-*- coding:utf-8 -*-
import numpy
import matplotlib.pyplot as plt

#读入一些数据，和标签，并且返回数据和标签
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))                        #获得数据集中每一个有多少个数据
    dataMat = []                                                                #初始化一个空的列表，用于存储数据集
    labelMat = []                                                               #初始化一个空的列表，用于存储标签集
    fr = open(fileName)                                                         #打开文件
    for line in fr.readlines():                                                 #从文件中读取每一行
        lineArr = []                                                            #初始化列表，用于缓存
        curLine = line.strip().split('\t')                                      #数据分割
        for i in range(numFeat - 1):                                            #轮询每一行数据
            lineArr.append(float(curLine[i]))                                   #最后一个数据是标签，其他是特征
        dataMat.append(lineArr)                                                 #生成数据集
        labelMat.append(float(curLine[-1]))                                     #生成标签集

    return dataMat,labelMat



#用于测试是否有某个值小于或者大于我们正在测试的阈值
#所有在阈值一边的数据会分类到类别-1，而在另一边的数据会被分类到+1
#dataMatrix     数据集
#dimen          特征
#threshVal      阈值
#threshIneq     比较方式
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArrray = numpy.ones((numpy.shape(dataMatrix)[0],1))                      #将数组的全部元素设置为+1
    #将不满足不等式要求的元素设置为-1
    if threshIneq == "lt":
        retArrray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArrray[dataMatrix[:,dimen] > threshVal] =  -1.0

    return retArrray


#在一个加权数据集中循环，找到具有最低错误率的单层决策树
#dataArr      数据集
#classLabels  标签集
#D            权重
def buildStump(dataArr,classLabels,D):
    dataMatrix = numpy.mat(dataArr)                                              #将数据集转化为矩阵
    labelMat = numpy.mat(classLabels).T                                          #将标签集转化为矩阵，并且转置
    m,n = numpy.shape(dataMatrix)                                                #获得数据集的行、列数据
    numSteps = 10.0                                                              #初始化步长为10，越大则步长越长，分类准确度越高，但运算次数也越多
    bestStump = {}                                                               #字典用于保存每个分类器的信息
    bestClasEst = numpy.mat(numpy.zeros((m,1)))                                  #初始化最佳分类器参数为1的矩阵
    minError = numpy.inf                                                         #初始化最小误差minError为无穷大
    for i in range(n):                                                           #遍历每一个列特征，找到差错最小时特征维数、阈值、以及比较符号
        rangeMin = dataMatrix[:,i].min()                                         #找到列特征的最小值
        rangeMax = dataMatrix[:,i].max()                                         #找到列特征的最大值
        stepSize = (rangeMax - rangeMin) / numSteps                              #（大-小）/ 分割数 = 步长 得到最小值到最大值需要的每一段距离
        for j in range(-1,int(numSteps) + 1):                                    #遍历步长，获取最佳阈值
            for inequal in ['lt','gt']:                                          #在大于和小于之间切换
                threshVal = (rangeMin + float(j) * stepSize)                     #最小值+次数*步长  每一次从最小值走的长度
                #获取预测标签列表
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)    #最有预测目标值，用于与目标值比较得到误差
                #获取当前列中那些行（1）预测出错，设置1表示出错便于后续量化计算差错权重
                errArr = numpy.mat(numpy.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr                                     #计算差错权重，计算当前分类方式整体错误率，找出最低错误率

                print("split:dim %d,thresh %.2f ,thresh ineqal:%s,the weighted error is %.3f" % (i,threshVal,inequal,weightedError))

                if weightedError < minError:                                     #选出最小错误的那个特征
                    minError = weightedError                                     #最小误差，后面用于更新D权值
                    bestClasEst = predictedVals.copy()                           #最优化预测值
                    bestStump['dim'] = i                                         #特征
                    bestStump['thresh'] = threshVal                              #到最小值的距离
                    bestStump['ineq'] = inequal                                  #大于还是小于，最有距离为-1

    return bestStump,minError,bestClasEst


#基于单层决策树的AdaBoost训练过程
'''
dataArr       数据集
classLabels   类别标签集
numIt         迭代次数   可以用户制定
任意分类器都可以作为基分类器
'''
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []                                                            #初始化最佳单层决策树组
    m = numpy.shape(dataArr)[0]                                                  #获得数据的数量
    D = numpy.mat(numpy.ones((m,1))/m)                                           #初始化权重D 为1/m
    aggClassEst = numpy.mat(numpy.zeros((m,1)))                                  #初始化累计类别估计值

    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)             #寻找最佳的单层决策树
        print("D:",D.T)
        #计算alpha  max(error,1e-16)用于确保在没有错误时不会发生除零溢出
        alpha = float(0.5*numpy.log((1.0 - error) / max(error,1e-16)))           #alpha的计算公式 alpha = 1/2 * ln((1-e)/e)
        bestStump['alpha'] = alpha                                               #记录该决策树的alpha值
        weakClassArr.append(bestStump)                                           #将最佳单层决策树加入到单层决策树组
        print("classEst:",classEst.T)
        #更新样本的权重 D = (D * exp(alpha))/Sum(D)
        #这里需要说明一下，对于classLabels表示数据应该属于哪一类，classEst表示预测的分类结果，两者都为-1，1的list
        #如果分对了，两者相乘结果为1，相反为-1.正好符合提到的公式
        expon = numpy.multiply(-1 * alpha * numpy.mat(classLabels).T,classEst)   #计算样本是否被正确分类的alpha值
        D = numpy.multiply(D,numpy.exp(expon))
        D = D / D.sum()

        #更新累计类别估计值
        aggClassEst += alpha * classEst
        print("aggClassEst:",aggClassEst.T)
        print(numpy.sign(aggClassEst))
        print(numpy.mat(classLabels).T)
        print(numpy.sign(aggClassEst) != numpy.mat(classLabels).T)
        #调用numpy.sign()函数得到二值分类结果，{x = 1，x > 0;x = 0,x = 0;x = -1,x < 0}
        #应为该集成方法是模型累加式的，所以需要累加各模型的估计值之和，如果最终估计值符号等于真实值符号，表明分队
        #这里!=，的意思是预测和真实值两个不相等就是True（1）.相等就是False（0），错误就是，1乘以任何数等于原数
        aggErrors = numpy.multiply(numpy.sign(aggClassEst) != numpy.mat(classLabels).T,numpy.ones((m,1)))
        errorRate = aggErrors.sum() / m                                          #计算累计错误率（总错误率）
        print("total error:",errorRate,"\n")
        if errorRate == 0.0:                                                     #如果错误率等于0.0，则退出循环
            break
    return weakClassArr,aggClassEst

#AdaBoost分类函数
#dataToClass    测试集
#classifierArr  弱分类器集
#每个弱分类器的结果以其对应的alpha值作为权重，所有这些弱分类器的结果加权和就是最后的结果
def adaClassify(datToClass,classifierArr):
    dataMatrix = numpy.mat(datToClass)          #将测试集转成矩阵
    m = numpy.shape(dataMatrix)[0]              #计算测试集的个数m
    aggClassEst = numpy.mat(numpy.zeros((m,1))) #初始化m个累加估计值
    for i in range(len(classifierArr)):         #遍历每一个弱分类器
        #对每个弱分类器进行类别的估计值计算
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]["thresh"],classifierArr[i]['ineq'])
        #估计值乘以权重作为单个分类器预测值，对分类器预测值进行累加计算累加估计值
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)

    return numpy.sign(aggClassEst)       #进行二分类求值


#ROC曲线绘制和AUC计算函数
#首先将分类样例按照其预测强度排序，先从排名最低的样例开始，所有排名更低的样例都被判定为反例
#而所有排名更高的样例都被判定为正例，该情况的对应点为<1.0,1.0>,然后将其移动到排名次低的样例中去，
#如果该样例属于正例，那么对真阳率进行修改，如果该样例属于反例，那么对假阴率进行修改
#predStrengths  表示分类器的预测强度
#classLabels    标签集
def plotROC(predStrengths,classLabels):
    #该元组保留的是绘制光标的位置
    cur = (1.0,1.0)                                                        #创建一个浮点数二元组，初始化为(1.0,1.0)
    ySum = 0.0                                                             #初始化ySum，用于计算AUC的值
    numPosClas = numpy.sum(numpy.array(classLabels) == 1.0)                #计算正例的数目，并且保存在numPosClas中

    yStep = 1 / float(numPosClas)                                          #确定了在y坐标轴上的步进数目  步长是 1/numPosClas
    xStep = 1 / float(len(classLabels) - numPosClas)                       #确定了在x坐标轴上的步进数目
    #argsort()函数返回的是数组值从小到大的索引值
    sortedIndicies = predStrengths.argsort()                               #获得排序索引
    fig = plt.figure()                                                     #获得绘图实例
    fig.clf()                                                              #清除所有轴，但是窗口打开
    ax = plt.subplot(111)                                                  #创建画布1行1列，第一个
    #从点<1.0,1.0>开始绘，一直到<0,0>  tolist()方法将矩阵转换成列表
    #当遍历表时，每得到一个标签为1.0的类，则要沿着y轴方向下降一个步长，即不断降低真阳率
    #对于每个其他类别的标签，则在x轴方向上倒退一个步长（假阴率方向）
    #只关注1这个类别标签，因此就无所谓是采用1/0标签还是+1/-1标签
    for index in sortedIndicies.tolist()[0]:                               #在所有排序值上进行循环
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep                                                   #下降一个步长
        else:
            delX = xStep                                                   #倒退一个步长
            delY = 0
            #为了计算AUC值，我们需要对多个小矩形的面积进行累加，小矩形的宽度是xStep，因此可以先对所有矩形的高度进行累加
            #最后再乘以xStep得到其总面积，所有高度的和（ySum）随着x轴的每次移动而渐次增加
            ySum += cur[1]
        ax.plot([cur[0],cur[0] - delX],[cur[1],cur[1] - delY],c = 'b')    #画线
        #更新当前点cur
        cur = (cur[0] - delX,cur[1] - delY)
    ax.plot([0,1],[0,1],'b--')                                            #画对角的虚线
    plt.xlabel("False Positive Rate")                                     #x坐标标签
    plt.ylabel("True Positive Rate")                                      #y坐标标签
    plt.title("ROC curve for AdaBoost Horse Colic Detection System")      #图像标题
    ax.axis([0,1,0,1])                                                    #画坐标轴
    plt.show()                                                            #显示图片
    print("the Area Under the Curve is:",ySum * xStep)                    #计算和输出AUC值，


if __name__ == "__main__":
    #获取数据集和标签
    datMat,classLabels = loadDataSet("horseColicTraining2.txt")
    #进行训练，迭代次数大于等于弱分类器个数
    classifierArr,aggClassEst = adaBoostTrainDS(datMat,classLabels,500)
    #绘制ROC曲线
    plotROC(aggClassEst.T,classLabels)
    #测试，其中classifierArr是弱分类器集合
    #读取测试集和标签
    testArr,testLabelArr = loadDataSet("horseColicTest2.txt")
    #进行测试
    prediction10 = adaClassify(testArr,classifierArr)
    errArr = numpy.mat(numpy.ones((67,1)))
    #计算错误率   分错的个数/总个数
    errRate = (errArr[prediction10 != numpy.mat(testLabelArr).T].sum()) / 67
    print(errRate)
