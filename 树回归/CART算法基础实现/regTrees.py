import numpy

#读取数据并且进行分割整理
def loadDataSet(fileName):
    dataMat = []                                              #创建一个数据集列表
    fr = open(fileName)                                       #打开文件
    for line in fr.readlines():                               #逐行读取文件内容
        curLine = line.strip().split('\t')                    #去掉前后空格，并且以\t进行分割
        fltLine = list(map(float,curLine))                    #map是一个高阶函数，将数据转换成float类型
        dataMat.append(fltLine)                               #将转换后的数据添加到数据集中
    return dataMat                                            #返回数据集

#创建叶节点函数
#当确定不在对数据进行切分时，将调用该函数来得到叶子节点
def regLeaf(dataSet):
    return numpy.mean(dataSet[:,-1])                          #在回归树模型中，此处为目标变量的均值

#误差计算函数
#计算目标变量的总方差
def regErr(dataSet):
    #numpy.var()是numpy的计算均方差的函数
    return numpy.var(dataSet[:,-1]) * numpy.shape(dataSet)[0]     #均方差*样本个数=总方差


#切分数据函数
#参数：数据集合，带切分的特征，该特征的某个值
#在给定特征和特征值的情况下，该函数通过数组过滤的方式将上述数据集合切分得到两个子集并返回
def binSplitDataSet(dataSet,feature,value):
    #简而言之就是取feature列大于value的第1个值所在行的第一个数据
    mat0 = dataSet[numpy.nonzero(dataSet[:,feature] > value)[0],:]
    #简而言之就是取feature列小于等于value的第1个值所在行的第一个数据
    mat1 = dataSet[numpy.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

#寻找最佳的二元切分方式函数   用最佳的方式切分数据集和生成相应的叶节点
#找到数据集切分的最佳位置
#回归树假设叶节点是常数值
#为成功构建以分段常数为叶节点的树，需要度量数据的一致性
#如何计算连续型数值的混乱度，首先计算所有数据的均值，然后计算每条数据的值到均值的差值
#为了对正负差值同等看待，使用绝对值或者平方值来代替差值
#dataSet   数据集
#leafType  创建叶节点的函数的引用
#errType   总方差计算函数的引用
#ops    参数元组
def chooseBestSplit(dataSet,leafType = regLeaf,errType=regErr,ops=(1,4)):
    #用于控制函数的停止时机
    tolS = ops[0]                                                                  #容许的误差下降值
    tolN = ops[1]                                                                  #切分的最少样本数
    #一旦停止切分，会生成一个叶节点
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:                                 #如果当前当前所有标签都相等，则表明分完，返回
        return None,leafType(dataSet)

    m,n = numpy.shape(dataSet)                                                     #获取当前数据集的结构
    #该误差S用于与新切分误差进行比对，来检查新切分能否降低误差值
    S = errType(dataSet)                                                           #计算当前数据集的总误差
    bestS = numpy.inf                                                              #将当前最佳误差设为无穷大
    bestIndex = 0                                                                  #初始化记录最佳的切分索引
    bestValue = 0                                                                  #初始化记录最佳的切分值
    #遍历所有特征及其可能的取值来找到使误差最小化的切分阈值
    for featIndex in range(n-1):                                                   #轮询每一个特征
        for splitVal in set((dataSet[:,featIndex].T.A.tolist())[0]):                                 #轮询每一个特征值
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)                #将数据切分成两份
            if(numpy.shape(mat0)[0] < tolN) or (numpy.shape(mat1)[0] < tolN):      #如果切分出的数据集很小则继续轮询下一个特征值
                continue
            newS = errType(mat0) + errType(mat1)                                   #计算切分误差
            #如果误差小于当前最小误差，那么将当前切分设定为最佳切分并且更新最小误差
            if newS < bestS:
                bestIndex = featIndex                                              #记录当前最佳切分索引
                bestValue = splitVal                                               #记录当前最佳切分值
                bestS = newS                                                       #记录当前最小误差
    #如果切分数据集后效果提升不大，那么就不进行切分操作，而直接创建叶节点
    if (S - bestS) < tolS:                                                         #如果误差减少太少，则返回
        return None,leafType(dataSet)
    #如果切分后的子集大小小于用于定义的参数tolN，也不进行切分
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)                       #将数据切分成两份
    if(numpy.shape(mat0)[0] < tolN) or (numpy.shape(mat1)[0] < tolN):              #如果切分出的数据集很小则退出
        return None,leafType(dataSet)

    return bestIndex,bestValue                                                     #返回当前最佳切分所有和特征值


#树构建函数
#dataSet  树构建函数
#leafType 建立叶节点函数
#errType  误差计算函数
#ops   其他参数
#该函数是一个递归函数
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    #参数val如果构建的是回归树，该参数是一个常数，如果构建的是模型树，该参数是一个线性方程
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)                     #切分数据为两部分
    if feat == None:                                                             #如果满足停止条件
        return val
    #如果不满足停止条件，将返回最佳切分索引和值
    retTree = {}
    retTree["spInd"] = feat                                                      #记录最佳切分索引
    retTree["spVal"] = val                                                       #记录最佳切分值
    lSet,rSet = binSplitDataSet(dataSet,feat,val)                                #寻找左子树和右子树
    retTree["left"] = createTree(lSet,leafType,errType,ops)                      #递归创建左子树
    retTree["right"] = createTree(rSet,leafType,errType,ops)                     #递归创建右子树
    return retTree


if __name__ == "__main__":
    myDat = loadDataSet('ex00.txt')
    myMat = numpy.mat(myDat)
    trees = createTree(myMat)
    print(trees)


# if __name__ == "__main__":
#     testMat = numpy.mat(numpy.eye(4))
#     print(testMat)
#     mat0,mat1 = binSplitDataSet(testMat,1,0.5)
#     print(mat0)
#     print(mat1)
