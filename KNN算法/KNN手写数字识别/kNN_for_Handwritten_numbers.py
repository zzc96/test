from numpy import *
import operator
from os import listdir

#inX 用于分类的输入向量
#dataSet 输入的训练样本集
#labels 标签向量
#k 选择最近邻的数目
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #矩阵相减
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    #差值平方
    sqDiffMat = diffMat**2
    #平方和 axis=1表示将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    #开平方
    distances = sqDistances**0.5
    #排序
    sortedDistIndicies = distances.argsort()
    classCount={}
    #找出前k个值
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #统计前k次标签出现的次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回最大次数的标签
    return sortedClassCount[0][0]

#将32*32的二进制图像矩阵转换为1*1024的向量
def img2vector(filename):
    #创建一个1024长度的矩阵，初始值为0
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        #先读取整行
        lineStr = fr.readline()
        for j in range(32):
            #将每一行的数据添加到矩阵的后面
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('./digits/trainingDigits')  #获取训练集的目录
    m = len(trainingFileList)                              #求训练集的总长度
    trainingMat = zeros((m, 1024))                         #创建训练矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]                  #获得文件名
        fileStr = fileNameStr.split('.')[0]                #去除后缀
        classNumStr = int(fileStr.split('_')[0])           #获得数字
        hwLabels.append(classNumStr)                       #标签集
        trainingMat[i, :] = img2vector('./digits/trainingDigits/%s' % fileNameStr) #数据集
    testFileList = listdir('./digits/testDigits')  #获得测试集
    errorCount = 0.0
    mTest = len(testFileList)                      #获取测试集的总长度
    for i in range(mTest):                         #循环测试
        fileNameStr = testFileList[i]              #获得文件名
        fileStr = fileNameStr.split('.')[0]        #去除后缀
        classNumStr = int(fileStr.split('_')[0])   #获得真实数字
        vectorUnderTest = img2vector('./digits/testDigits/%s' % fileNameStr)   #测试集归一化
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)  #预测
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0   #错误的情况
    print("\nthe total number of errors is: %d" % errorCount)    #错误总数
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))   #错误比例

if __name__ == "__main__":
    handwritingClassTest()