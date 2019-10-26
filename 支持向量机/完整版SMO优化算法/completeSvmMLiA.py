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

#optStruct是一个类，他的对象用于保存所有的重要值
#数据可以通过一个对象来传递
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler):                    #构造函数，实现成员变量的填充
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = numpy.shape(dataMatIn)[0]
        self.alphas = numpy.mat(numpy.zeros((self.m,1)))
        self.b = 0
        #eCache 用于误差缓存
        #eCache的第一列给出eCache是否有效的标志位
        #eCache的第二列给出的是实际的E值
        self.eCache = numpy.mat(numpy.zeros((self.m,2)))


#对于给定的alpha值，计算E值并且返回
#其实就是f(x) = ∑alpha_i*y_i*<x_i,x> + b
def calcEk(oS,k):
    # 计算预测的类别，该公式是分隔超平面的公式    .T表示转置
    fXk = float(numpy.multiply(oS.alphas,oS.labelMat).T * (oS.X * oS.X[k,:].T)) + oS.b
    #计算误差
    Ek = fXk - float(oS.labelMat[k])

    return Ek


#用于选择第二个alpha的值或者说内循环的alpha值
#这里的目标是选择合适的第二个alpha值以保证在每次优化中采用最大步长
def selectJ(i,oS,Ei):
    maxK = -1                                                             #定义使得计算E的差值最大的位置
    maxDeltaE = 0                                                         #初始化最大步长
    Ej = 0                                                                #初始化使得差值最大的Ej
    oS.eCache[i] = [1,Ei]                                                 #将输入值Ei在缓存中设置成有效的，有效意味着已经计算好了
    #nonzero()语句返回的是非零E值所对应的alpha值，不是E值本身
    #.A的意思是把矩阵转换成数组，Os.eCache[:,0].A,取的是一列的数据E值有效位，转换为一列
    validEcacheList = numpy.nonzero(oS.eCache[:,0].A)[0]                  #构建一个非零列表

    #如果长度大于0个，
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:                                         #索引每一个值
            if k == i:                                                    #如果相等，则表明两个alpha值一样，直接跳出本次循环，进行下一次循环
                continue
            Ek = calcEk(oS,k)                                             #计算Ek
            deltaE = abs(Ei - Ek)                                         #求两个E值得差值，选择最大得步长
            if(deltaE > maxDeltaE):                                       #比较，选择最大的差值
                maxK = k                                                  #标记位置
                maxDeltaE = deltaE                                        #记录最大值
                Ej = Ek                                                   #记录差值最大的Ek
        return maxK,Ej                                                    #返回
    else:
        #如果第一次，则E都为0，进行随机选择alpha的i,j，并且进行计算
        j = selectJrand(i,oS.m)
        Ej = calcEk(oS,j)

    return j,Ej


#计算误差并且存入缓存，在对alpha值进行优化之后用到该值
def updateEk(oS,k):
    #计算E值并且返回
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]                                                  #进行缓存

#完整版的SMO优化例程
def innerL(i,oS):
    Ei = calcEk(oS,i)                                                      #计算第一个alpha的差值
    #寻找非边界的alpha，为寻找第二个alpha做准备
    # 如果误差很大，可以对数据实例所对应的alpha值进行优化，正负间隔都会被测试，同时检查alpha值
    # 在该if语句中，同时检查alpha值，以保证其不能等于0或者C。由于后面alpha小于0或大于C时都将被调整为0或者C
    # 所以一旦在该if语句中他们等于这两个值得话，那么他们就已经在‘边界’上了，不能再增加或者减小，不值得在优化
    if((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i,oS,Ei)                                            #选择第二个alpha，并且得到Ej
        #保存旧的alpha，方便后面比较使用
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        # 用于讲alpha[j]调整到0-c之间，
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0,oS.alphas[j] - oS.alphas[i])                         #对SMO最优化问题的子问题的约束条件分析
            H = min(oS.C,oS.C + oS.alphas[j] - oS.alphas[i])               #L和H分别是alpha所在的对角线端点的界
        else:                                                              #调整alpha[j]位于0-c之间
            L = max(0,oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C,oS.alphas[j] + oS.alphas[i])

        if L == H:                                                         #L == H则停止本次循环
            print("L == H")
            return 0

        #eta是一个中间变量，eta=2Xi*Xi - XiXi - XjXj  是alphas[j]的最优修改量
        eta = 2.0 * oS.X[i,:] * oS.X[j,:].T - oS.X[i,:] * oS.X[i,:].T - \
            oS.X[j,:] * oS.X[j,:].T

        # 如果eta为0，那么没必要计算新的alphas[j]
        if eta >= 0:                                                       #eta >= 0停止本次循环，这里是简化计算
            print("eta >= 0")
            return 0

        # 计算出一个新的alphas[j]
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta                   #沿着约束方向未考虑不等式约束时的alphas[j]的解
        # 使用L和H值对其进行调整
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)                         #此处是考虑不等式约束的alphas[j]解
        updateEk(oS,j)                                                     #将选出的第二个alpha对应的E添加到eCache
        if(abs(oS.alphas[j] - alphaJold) < 0.00001):                       #检查alphas[j]是否有轻微改变
            print("j not moving enough")                                   #如果alphas值不再变化，就停止该alpha的优化
            return 0

        # 更新另一个alpha，即第一个alpha
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        #添加到缓存eCache
        updateEk(oS,i)
        # 完成alpha变量的更新后，都要重新计算阈值b
        # 计算常数项b
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
            oS.X[i,:] * oS.X[i,:].T - oS.labelMat[j] * \
             (oS.alphas[j] - alphaJold) * oS.X[i,:] * oS.X[j,:].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
            oS.X[i,:] * oS.X[j,:].T - oS.labelMat[j] * \
             (oS.alphas[j] - alphaJold) * oS.X[j,:] * oS.X[j,:].T

        #根据条件更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0                                          #alphas[i]和alphas[j]是0或者C，就取中点作为b
        return 1
    else:
        return 0

#外循环即第一个alpha的选择
def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup = ("lin",0)):
    #把数据传入数据结构中
    oS = optStruct(numpy.mat(dataMatIn),numpy.mat(classLabels).transpose(),C,toler)
    iter = 0                                                                #初始化迭代次数
    entireSet = True                                                        #整个开始启动的标记，true为全部，false为边界
    alphaPairsChanged = 0                                                   #alpha对变化标记
    #执行while的条件是：
    #1：迭代次数达到最大
    #2：alpha值存在变化，或者整个开始启动即entireSet = True
    #这里的一次迭代定义为一次循环过程
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):                                           #在数据集上遍历任意可能的alpha值
                #进入innerL时第一个alpha选取就为1
                alphaPairsChanged += innerL(i,oS)                           #调用innerL()来选择第二个alpha
                #第二个则根据最大的E差值进行选择
                #如果寻找满足条件则返回1，否则返回0，alphaPairsChanged是记录变化的alpha的变化次数
            print("fullSet,iter:%d i:%d,pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1                                                       #整个遍历完成之后，alpha更新结束 为一次迭代
        else:                                                               #遍历非边界的alpha值，也就是不在0或者C上的值
            #寻找非边界的点
            nonBoundIs = numpy.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:                                            #遍历
                alphaPairsChanged += innerL(i,oS)
                print("non-bound,iter:%d i:%d,pair changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        #在非边界循环和完整遍历之间进行切换
        if entireSet:
            entireSet = False       #开始时先遍历整个alpha，然后迭代完成一次后，开始转向非边界进行遍历
        elif (alphaPairsChanged == 0):
            entireSet = True        #如果遍历完所有的非边界都没有可更新的alpha，则继续转为全局遍历
        #大于迭代次数
        print("iteration number :%d" % iter)
    #返回常数b和alpha值
    return oS.b,oS.alphas

#根据alpha的值计算w值
def clacWs(alphas,dataArr,classLabels):
    X = numpy.mat(dataArr)                                           #将数据转换成矩阵
    labelMat = numpy.mat(classLabels).transpose()                    #将标签转换成矩阵并且转置
    m,n = numpy.shape(X)                                             #获得数据的形状
    w = numpy.zeros((n,1))                                           #存在n个特征，就初始化n个为0的w
    # w=求和(ai*yi*xi),求和对象是支持向量，xi，yi为支持向量对应的label和data
    for i in range(m):                                               #轮询alphas
        w += numpy.multiply(alphas[i] * labelMat[i],X[i,:].T)
    return w                                                         #返回w值

if __name__ == "__main__":
    dataArr,labelArr = loadDataSet("./testSet.txt")
    b,alphas = smoP(dataArr,labelArr,0.6,0.001,40)

    #寻找支持向量
    plotPoint = []
    for i in range(100):
        if alphas[i] > 0.0:
            plotPoint.append(dataArr[i])

    # 计算weight
    w1 = clacWs(alphas,dataArr,labelArr)
    print("b and w:",b,w1)

    # 数组过滤，值对numpy类型有作用，对python的正则表没有任何作用
    # alphas > 0得到一个布尔数组，将数组放到原始的矩阵中，就会得到一个numpy矩阵
    alphas = alphas[alphas > 0]
    #画图
    plotSupportVectors.plotDrawPoint(plotPoint,b.tolist()[0][0],w1.tolist())

    #使用计算好的数据进行分类
    datMat = numpy.mat(dataArr)
    ones = datMat[0]*numpy.mat(w1) + b
    print("ones:",ones)