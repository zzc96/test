#-*- coding:utf-8 -*-
from numpy import *

#创建了一些实验样本
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]

    classVec = [0,1,0,1,0,1]   #1 代表侮辱性文字，0代表正常言论
    #postingList是进行词条切分后的文档集合，
    #classVec是类别标签的集合
    return postingList,classVec

#创建一个包含所有文档中出现的不重复的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)               #set返回不重复的词表 | 为包含两个集合中的所有元素，求两个集合的并集

    return list(vocabSet)                                 #将集合转换成列表


#输入的参数为词汇表以及某个文档，
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)                        #生成len(vocabList)个0元素的列表
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1          #如果出现了词汇表中的单词，将输出的文档向量中的对应值设为1
        else:
            print("the word: %s is not in my Vocabulary!"%word)
    #输出文档向量  向量元素为1表明单词在词汇表中出现，为0表明单词在词汇表中没有出现
    return returnVec


#文档矩阵 trainMatrix
#文档类别标签构成的向量trainCategory
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                      #求取文档矩阵的行数，就是类别总数
    numWords = len(trainMatrix[0])                       #求取文档矩阵的列数
    pAbusive = sum(trainCategory)/float(numTrainDocs)    #计算文档属于侮辱类文档的概率
    #此处主要是为了预防计算p(w0|1)p(w1|1)p(w2|1)时，其中一个概率值为0，那么最后的成绩也为0，为降低这种影响，将所有词的出现次数初始化为1，分母初始化为2
    p0Num = ones(numWords)                               #初始化概率  0 非侮辱性 的分子
    p1Num = ones(numWords)                               #初始化概率  1 侮辱性 的分子
    p0Denom = 2.0                                        #初始化分母
    p1Denom = 2.0                                        #初始化分母
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                        #如果是侮辱性
            p1Num += trainMatrix[i]                      #矩阵相加
            p1Denom += sum(trainMatrix[i])
        else:                                            #如果不是侮辱性
            p0Num += trainMatrix[i]                      #矩阵相加
            p0Denom += sum(trainMatrix[i])               #总数

    #该处使用log主要是为了预防下溢出，应为大部分p(w|c)因子都很小，会导致下溢出
    #取自然对数能有效避免，再数学中，采用自然对数不会有任何损失
    p1Vect = log(p1Num/p1Denom)                               #每个元素除以类别中的总词数
    p0Vect = log(p0Num/p0Denom)

    return p0Vect,p1Vect,pAbusive


#要分类的向量，以及使用函数trainNB0计算得到的概率
#类别0的概率向量   类别1的概率向量
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    #在代数中有ln(a*b) = ln(a) + ln(b)
    #vec2Classify是词向量对应测试所存在与否的列表，乘以原本对应的概率再相加，就得到测试的概率，在用ln(a*b) = ln(a) + ln(b)
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)                  #向量相乘，词汇表中所有词汇对应值相加，然后加到类别的对数概率上
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)            #二分类问题，概率总和为1
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    print(p0V, p1V, pAb)

    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(thisDoc)
    print(testEntry,"classified as:",classifyNB(thisDoc,p0V,p1V,pAb))

    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print(thisDoc)
    print(testEntry,"classified as:",classifyNB(thisDoc,p0V,p1V,pAb))

if __name__ == "__main__":
    testingNB()

# if __name__ == "__main__":
#     listOPosts,listClasses = loadDataSet()
#     myVocabList = createVocabList(listOPosts)
#     print(myVocabList)
#     returnVec = setOfWords2Vec(myVocabList,listOPosts[0])
#     print(returnVec)
#     returnVec = setOfWords2Vec(myVocabList, listOPosts[3])
#     print(returnVec)

# if __name__ == "__main__":
#     listOPosts,listClasses = loadDataSet()
#     myVocabList = createVocabList(listOPosts)
#     print(myVocabList)
#     trainMat = []
#     for postinDoc in listOPosts:
#         trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
#
#     p0V,p1V,pAb = trainNB0(trainMat,listClasses)
#     print(p0V,p1V,pAb)

