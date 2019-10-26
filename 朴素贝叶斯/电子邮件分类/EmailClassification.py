#-*- coding:utf-8 -*-
from numpy import *
import re

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
        if word in vocabList:                             #采用词集模型
            returnVec[vocabList.index(word)] = 1          #如果出现了词汇表中的单词，将输出的文档向量中的对应值设为1
        else:
            print("the word: %s is not in my Vocabulary!"%word)
    #输出文档向量  向量元素为1表明单词在词汇表中出现，为0表明单词在词汇表中没有出现
    return returnVec


#文档矩阵 trainMatrix
#文档类别标签构成的向量trainCategory
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)                      #求取文档矩阵的行数
    numWords = len(trainMatrix[0])                       #求取文档矩阵的列数
    pAbusive = sum(trainCategory)/float(numTrainDocs)    #计算文档属于侮辱类文档的概率
    p0Num = ones(numWords)                              #初始化概率  0 非侮辱性 的分子
    p1Num = ones(numWords)                              #初始化概率  1 侮辱性 的分子
    p0Denom = 2.0                                        #初始化分母
    p1Denom = 2.0                                        #初始化分母
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                        #如果是侮辱性
            p1Num += trainMatrix[i]                      #矩阵相加
            p1Denom += sum(trainMatrix[i])
        else:                                            #如果不是侮辱性
            p0Num += trainMatrix[i]                      #矩阵相加
            p0Denom += sum(trainMatrix[i])               #总数

    p1Vect = log(p1Num/p1Denom)                               #每个元素除以类别中的总词数
    p0Vect = log(p0Num/p0Denom)

    return p0Vect,p1Vect,pAbusive


#要分类的向量，以及使用函数trainNB0计算得到的概率
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)                  #向量相乘，词汇表中所有词汇对应值相加，然后加到类别的对数概率上
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)            #二分类问题，概率总和为1
    if p1 > p0:
        return 1
    else:
        return 0


def textParse(bigString):
    #\W 表示匹配任意不是字母、数字、下划线、汉字的字符
    #* 表示前面的字符可以出现零次或者多次
    listOfTokens = re.split(r'\W*',bigString)                           #正则表达式
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]        #采用列表生成式去清洗数据
    #去掉少于两个字符的字符串，并将所有字符串转换成小写


#对垃圾邮件分类器进行自动化处理
def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        #将所有邮件解析为词列表
        wordList = textParse(open('./email/spam/%d.txt' % i).read())    #导入垃圾邮件下的分类
        docList.append(wordList)                                        #初始词列表
        fullText.extend(wordList)
        classList.append(1)                                             #垃圾邮件标定为1

        wordList = textParse(open('./email/ham/%d.txt' % i).read())     #导入正常邮件
        docList.append(wordList)                                        #初始列表
        fullText.extend(wordList)
        classList.append(0)                                             #正常邮件标定为0

    vocabList = createVocabList(docList)                                #去重，构建词集

    #随机构建测试集和训练集   留存交叉验证的过程
    trainingSet = list(range(50))                                       #邮件数是50  0-49
    testSet = []
    for i in range(10):                                                 #随机选择10份邮件作为测试集
        randIndex = int(random.uniform(0,len(trainingSet)))             #从0-49的数字中随机选取一个数
        testSet.append(trainingSet[randIndex])                          #加入测试集列表
        del(trainingSet[randIndex])                                     #在训练集列表中删除

    #根据训练集构建词集
    trainMat = []                                                       #创建词向量列表
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))    #构建词向量  在trainNB0()函数中用于计算分类所需的概率
        trainClasses.append(classList[docIndex])                        #类别

    #模型训练
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0

    #模型评估
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])                    #计算测试词集
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:      #计算概率得出结果并且进行比对
            errorCount += 1                                                         #计算错误数量

    print('the error rate is :',float(errorCount)/len(testSet))

    #测试使用
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(vocabList,testEntry))
    print(thisDoc)
    print(testEntry,"classified as:",classifyNB(thisDoc,p0V,p1V,pSpam))

if __name__ == "__main__":
    #模型训练和评估
    spamTest()