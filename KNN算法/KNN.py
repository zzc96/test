#导入numpy包用于科学计算
#导入operator包用于运算符计算
from numpy import *
import operator


#距离计算函数
#inX 用于分类的输入向量
#dataSet 输入的训练样本集
#labels 标签向量
#k 选择最近邻的数目
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]                                    #获得数据集的个数
    #差值计算，距离相减
    diffMat = tile(inX, (dataSetSize,1)) - dataSet                    #numpy中tile函数用于将inX，生成dataSetSize行1列的数据
    #差值平方
    sqDiffMat = diffMat**2
    #平方和 axis=1表示将数据的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    #开平方
    distances = sqDistances**0.5                                      #0.5次方相当于开平方
    #排序   argsort()序号排序
    sortedDistIndicies = distances.argsort()                          #由低到高进行排序，此处得到的数据排序后所在的序号
    classCount={}                                                     #初始化统计字典
    #找出前k个值
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]                    #获得对应标签
        #统计前k次标签出现的次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1     #此处字典的get()函数用于查找制定的健所对应的值，如果健不存在，则返回default，此处为0
    #排序  sorted对所有可迭代的对象进行排序操作，返回一个新的列表
    #第一个参数是可迭代对象，key是用来进行比较的元素，制定可迭代对象中的一个元素来进行排序    reverse=True为降序（排序规则）
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回最大次数的标签
    return sortedClassCount[0][0]

#创建数据集和标签
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])                       #生成数据集
    labels = ['A','A','B','B']                                               #生成标签
    return group, labels

if __name__ == "__main__":
    group,labels = createDataSet()
    sort = classify0([0,0],group,labels,3)
    print(sort)