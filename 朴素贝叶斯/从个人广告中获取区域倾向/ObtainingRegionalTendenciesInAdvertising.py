#-*- coding:utf-8 -*-
from numpy import *
import re
import operator
import feedparser

#创建一个包含所有文档中出现的不重复的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)               #set返回不重复的词表 | 为包含两个集合中的所有元素，求两个集合的并集

    return list(vocabSet)                                 #将集合转换成列表


#输入的参数为词汇表以及某个文档，
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0]*len(vocabList)                        #生成len(vocabList)个0元素的列表
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1          #如果出现了词汇表中的单词，将输出的文档向量中的对应值加1
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


#计算出现的概率
def calcMostFreq(vocabList,fullText):
    freqDict = {}                                                       #申明空字典
    for token in vocabList:
        freqDict[token] = fullText.count(token)                         #遍历每个词，并且寻找每个出现的次数，且记录在字典中
                                                                        #根据出现次数从高到低对字典进行排序
    sortedFreq = sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)

    return sortedFreq[:30]                                              #返回排序最高的30个单词


def localWords(feed1,feed0):
    docList = []
    classList = []
    fullText = []

    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):                                          #构建词汇表
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)                              #去重，创建词汇表
    top30Words = calcMostFreq(vocabList,fullText)                     #返回30个排序最高的单词
    for pairW in top30Words:                                          #移除排序最高的30个单词  移除高频词
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    print("minLen:",minLen)
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(20):                                                #制作测试集
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:                                       #制作训练集
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))      #训练
    errorCount = 0
    #模型评估
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1

    print('the error rate is :',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    vocabList,p0V,p1V = localWords(ny,sf)
    topNY = []
    topSF = []

    for i in range(len(p0V)):
        if p0V[i] > -6.0 :                                            #取大于-6.0这个阈值的值
            topSF.append((vocabList[i],p0V[i]))                       #列表中嵌套元组
        if p1V[i] > -6.0:
            topNY.append((vocabList[i],p1V[i]))

    sortedSF = sorted(topSF,key=lambda pair:pair[1],reverse=True)     #排序  以列表中单项元组的第1（下标1，实际2）个作为排序标准
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")

    for item in sortedSF:                                             #输出
        print(item[0])

    #按照条件概率进行排序
    sortedNY = sorted(topNY,key=lambda pair:pair[1],reverse=True)     #排序  以列表中单项元组的第1（下标1，实际2）个作为排序标准
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


if __name__ == "__main__":
    # ny = feedparser.parse('http://newyork.craiglist.org/stp/index.rss')
    ny = feedparser.parse("http://www.nasa.gov/rss/dyn/image_of_the_day.rss")
    # sf = feedparser.parse('http://sfbay.craiglist.org/stp/index.rss')
    sf = feedparser.parse("http://sports.yahoo.com/nba/teams/hou/rss.xml")
    print(ny,sf)
    getTopWords(ny,sf)

#扩展，可以移除固定停用词