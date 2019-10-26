#-*- coding:utf-8 -*-
import random
import numpy
import plotSupportVectors

#获取数据集和标签
def loadDataSet(fileName):
    dataMat = []                                                #数据集列表
    labelMat = []                                               #标签列表
    fr = open(fileName)                                         #打开数据集文件

    for line in fr.readlines():                                 #逐行读取数据
        lineArr = line.strip().split('\t')                      #去除头尾空格，并且以'\t'进行数据分割
        dataMat.append([float(lineArr[0]),float(lineArr[1])])   #获取两个特征的数据，并且加入数据集
        labelMat.append(float(lineArr[2]))                      #获取特征，加入特征集

    return dataMat,labelMat                                     #返回数据集和特征集


#i是第一个alpha的下标
#m是所有alpha的数目
def selectJrand(i,m):
    j = i
    while(j == i):                                              #只要函数值不等于输入值i，函数就会进行随机选择
        j = int(random.uniform(0,m))                            #从0-m中随机选择

    return j                                                    #返回选择的下标


#用于调整alpha的值
def clipAlpha(aj,H,L):
    if aj > H:                                                  #如果alpha大于H，则等于H
        aj = H
    if L > aj:                                                  #如果alpha小于L，则等于L
        aj = L

    return aj

#SMO算法计算函数
'''
dataMatIn       数据集
classLabels     类别标签
C               常数C
toler           容错率
maxIter         推出前最大的循环次数
'''
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = numpy.mat(dataMatIn)                                    #将数据集转换成NumPy矩阵
    labelMat = numpy.mat(classLabels).transpose()                        #将类别转换成NumPy矩阵，并且进行转置得到一个标签的列向量
    b = 0                                                                #初始化偏执
    m,n = numpy.shape(dataMatrix)                                        #求取数据集的维度  得到行、列数
    alphas = numpy.mat(numpy.zeros((m,1)))                               #创建alpha向量并且初始化为0向量，并且构建了一个alpha列矩阵
    iter = 0                                                             #迭代次数
    while(iter < maxIter):                                               #如果当前迭代次数小于最大迭代次数
        alphaPairsChanged = 0                                            #用于记录alpha是否已经优化
        for i in range(m):                                               #对整个集合顺序遍历
            #计算预测的类别，该公式是分隔超平面的公式    .T表示转置
            fXi = float(numpy.multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])                                    #计算误差
            #如果误差很大，可以对数据实例所对应的alpha值进行优化，正负间隔都会被测试，同时检查alpha值
            #在该if语句中，同时检查alpha值，以保证其不能等于0或者C。由于后面alpha小于0或大于C时都将被调整为0或者C
            #所以一旦在该if语句中他们等于这两个值得话，那么他们就已经在‘边界’上了，不能再增加或者减小，不值得在优化
            if((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)                                         #随机选择不等于i的0-m的第二个alpha值
                #计算预测的第二个类别，该公式是分隔超平面的公式
                fXj = float(numpy.multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])                                #计算误差
                alphaIold = alphas[i].copy()                                 #缓存老数据，便于新老数据进行比较
                alphaJold = alphas[j].copy()                                 #缓存老数据
                #用于讲alpha[j]调整到0-c之间，
                if(labelMat[i] != labelMat[j]):                              #对SMO最优化问题的子问题的约束条件分析
                    L = max(0,alphas[j] - alphas[i])                         #L和H分别是alpha所在的对角线端点的界
                    H = min(C,C + alphas[j] - alphas[i])                     #调整alpha[j]位于0-c之间
                else:
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,alphas[j] + alphas[i])
                if L == H:                                                   #L == H则停止本次循环
                    print("L==H")
                    continue
                #eta是一个中间变量，eta=2Xi*Xi - XiXi - XjXj  是alphas[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - \
                    dataMatrix[i,:] * dataMatrix[i,:].T - \
                    dataMatrix[j,:] * dataMatrix[j,:].T
                #如果eta为0，那么没必要计算新的alphas[j]
                if eta >= 0:                                                 #eta >= 0停止本次循环，这里是简化计算
                    print("eta>=0")
                    continue

                #计算出一个新的alphas[j]
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta                     #沿着约束方向未考虑不等式约束时的alphas[j]的解
                #使用L和H值对其进行调整
                alphas[j] = clipAlpha(alphas[j],H,L)                         #此处是考虑不等式约束的alphas[j]解
                if (abs(alphas[j] - alphaJold) < 0.00001):                   #检查alphas[j]是否有轻微改变
                    print("j not moving enough")                             #如果alphas值不再变化，就停止该alpha的优化
                    continue
                # 更新alphas[i]  alphas[i]和alphas[j]同样进行改变，改变的大小一样，但是方向正好相反，一个增加，一个减少
                alphas[i] += labelMat[j] * labelMat[i] * \
                            (alphaJold - alphas[j])
                #完成两个alpha变量的更新后，都要重新计算阈值b
                #计算常数项b
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                    dataMatrix[i,:] * dataMatrix[i,:].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[i,:] * dataMatrix[j,:].T

                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                    dataMatrix[i,:] * dataMatrix[j,:].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[j,:] * dataMatrix[j,:].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif(0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0                                         #alphas[i]和alphas[j]是0或者C，就取中点作为b
                alphaPairsChanged += 1                                        #到此说明已经成功改变了一对alphas
                print("iter:%d i:%d,pair changed %d" % (iter,i,alphaPairsChanged))

        #检查alpha是否更新，如果alpha有更新则将iter设置为0后继续运行程序
        if (alphaPairsChanged == 0):                                      #如果alphas不再改变迭代次数就加1
            iter += 1
        else:
            iter = 0

        print("iteration number : %d" % iter)
    #在所有数据集上遍历maxIter次，且不再发生任何alpha修改之后，程序停止并且退出while循环
    return b,alphas


#根据alpha值计算w值
def calcWeight(data,label,alphas):
    dataMatrix = numpy.mat(data)                                     #将数据转换成矩阵
    labelMatrix = numpy.mat(label).transpose()                       #将标签转换成矩阵并且转置
    m,n = dataMatrix.shape                                           #获得数据的形状
    w = numpy.mat(numpy.zeros((1,n)))                                #存在n个特征，就初始化n个为0的w

    #w=求和(ai*yi*xi),求和对象是支持向量，即，ai>0的样本点，xi，yi为支持向量对应的label和data
    for i in range(m):                                               #轮询alphas
        if alphas[i] > 0.0:                                          #如果大于0，表明为支持向量
            w += labelMatrix[i] * alphas[i] * dataMatrix[i,:]

    return w.tolist()                                                #将numpy的矩阵转换成列表

if __name__ == "__main__":
    dataArr,labelArr = loadDataSet("./testSet.txt")
    print(dataArr,labelArr)
    b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)

    #寻找支持向量的点
    plotPoint = []
    for i in range(100):
        if alphas[i] > 0.0:
            plotPoint.append(dataArr[i])

    #计算weight
    w1 = calcWeight(dataArr,labelArr,alphas)
    print(w1)

    # 数组过滤，值对numpy类型有作用，对python的正则表没有任何作用
    # alphas > 0得到一个布尔数组，将数组放到原始的矩阵中，就会得到一个numpy矩阵
    alphas = alphas[alphas > 0]
    print(alphas)
    #作图
    plotSupportVectors.plotDrawPoint(plotPoint,b.tolist()[0][0],w1[0])