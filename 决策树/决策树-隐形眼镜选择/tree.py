from math import log
import operator
import pickle
import treePlotter
import os

#计算香农熵(经验熵)
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)                         #计算数据集中实例的总数
    labelCounts = {}                                  #创建一个数据字典，键值是最后一列的数值

    for featVec in dataSet:                           #遍历数据集
        currentLabel = featVec[-1]                    #键值是最后一列的数值
        if currentLabel not in labelCounts.keys():    #如果当前键值不存在
            labelCounts[currentLabel] = 0             #将当前键值加入字典 并且类别出现次数为0
        labelCounts[currentLabel] += 1                #如果存在类别，将当前类别值加1

    shannonEnt = 0.0
    for key in labelCounts:                           #遍历标签
        prob = float(labelCounts[key])/numEntries     #求每个类别出现的概率
        shannonEnt -= prob * log(prob,2)              #求香农熵  负数求和等于-=
    return shannonEnt                                 #返回香农熵

#读取隐形眼镜数据
def readDataSet(filename):
    fr = open(filename)
    lenses = [inst.strip().split("\t") for inst in fr.readlines()]
    lensesLabels = ['age','prescript','astigmatic','tearRate']

    return lenses,lensesLabels

#按照给定特征划分数据集
#当我们按照某个特征划分数据集时，就需要将所有符合要求的元素抽取出来
#参数：待划分的数据集，划分数据集的特征，需要返回的特征值（划分的特征值）
def splitDataSet(dataSet,axis,value):
    retDataSet = []                                     #新建一个list对象
    for featVec in dataSet:                             #轮询原始特征中的每个元素
        if featVec[axis] == value:                      #如果符合要求
            reducedFeatVec = featVec[:axis]             #顾头不顾尾，axis那一列不需要
            reducedFeatVec.extend(featVec[axis+1:])     #顾头不顾尾，axis那一列不需要
            retDataSet.append(reducedFeatVec)           #抽取的特征集合列表
    return retDataSet

#选取特征，划分数据集，计算得出最好的划分数据集的特征
#在函数中调用的数据需要满足的需求：
"""
1:数据必须是一种由列表元素组成的列表，并且所有的列表元素都具有相同的数据长度
2：最后一列或者每个实例的最后一个元素是当前实例的类别标签
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                       #计算当前数据集包含多少特征属性
    baseEntropy = calcShannonEnt(dataSet)                   #计算原始香农熵（类别）
    bestInfoGain = 0.0                                      #初始化信息增益
    bestFeature = -1
    for i in range(numFeatures):                            #遍历数据集中的所有特征
        featList = [example[i] for example in dataSet]      #列表推导，将数据集中的第i个特征写入list中
        uniqueVals = set(featList)                          #使用集合去重,获得唯一特征
        newEntropy = 0.0                                    #熵加权和-初始化
        for value in uniqueVals:                            #遍历当前特征中的所有唯一属性
            subDataSet = splitDataSet(dataSet,i,value)      #对每个特征划分一次数据集
            prob = len(subDataSet)/float(len(dataSet))      #计算特征概率
            newEntropy += prob * calcShannonEnt(subDataSet) #计算信息熵
        infoGain = baseEntropy - newEntropy                 #计算信息增益
        if (infoGain > bestInfoGain):                       #比较所有特征中的信息增益，返回最好特征划分的索引值
            bestInfoGain = infoGain                         #获得最好熵  信息增益越大越好
            bestFeature = i                                 #获得最好特征划分的索引值

    return bestFeature                                      #返回最好特征的索引值

"""
该函数使用分类名称的列表，创建键值为classList中唯一值的数据字典
字典对象存储了classList中每个类标签出现的频率，最后利用operator
操作键值排序字典，并返回次数最多的分类名称
"""
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():                   #判定当前类别是否在字典中
            classCount[vote] = 0                            #如果不在，那么就创建键值，并且初始化为0
        classCount[vote] += 1                               #如果存在，那么数据就加1
    #对类别进行排序，排降序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #返回次数最多的类别
    return sortedClassCount[0][0]


"""创建决策树"""
#参数：数据集和标签列表
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]         #取得数据集的所有类标签
    if classList.count(classList[0]) == len(classList):      #如果所有类标签都相同，则返回
        return classList[0]
    if len(dataSet[0]) == 1:                                 #使用完了所有特征，仍不能将数据集划分成仅包含唯一类别的分组
        return majorityCnt(classList)                        #返回出现次数最多的类别
    bestFeat = chooseBestFeatureToSplit(dataSet)             #获得最好特征划分的索引值
    bestFeatLabel = labels[bestFeat]                         #根据索引找到对应的特征名称
    myTree = {bestFeatLabel:{}}                              #使用字典变量存储树的所有信息
    del(labels[bestFeat])                                    #删除描述
    featValues = [example[bestFeat] for example in dataSet]  #根据索引值获得最好特征划分列的数据
    uniqueVals = set(featValues)                             #使用字典使列表唯一
    for value in uniqueVals:
        subLabels = labels[:]                                #复制类标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
                                                             #递归，得到的返回值插入字典中
    return myTree

#保存决策树
#使用pickle序列化对象，并且将序列化对象存储在磁盘
def storeTree(inputTree,filename):
    fw = open(filename,"wb")                                  #打开文件
    pickle.dump(inputTree,fw)                                #序列话对象并且存储
    fw.close()

#读取决策树内容
def grabTree(filename):
    fr = open(filename,"rb")                                      #打开文件
    treeData = pickle.load(fr)                                   #读取数据并且序列话
    fr.close()
    return treeData


if __name__ == "__main__":
    myDat,labels = readDataSet("./lenses.txt")
    myTree = createTree(myDat,labels)
    print(myTree)
    treePlotter.createPlot(myTree)
    storeTree(myTree,"./classifierStorage.txt")
    # gTree = grabTree("./classifierStorage.txt")
    # print(gTree)
