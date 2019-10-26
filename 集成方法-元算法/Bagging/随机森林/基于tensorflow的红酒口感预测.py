# -*- coding:utf-8 -*-
import urllib.request
import numpy
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import pylab as plot

#从网页中读取数据
url="http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data=urllib.request.urlopen(url)

#将数据中第一行的属性读取出来放在names列表中，将其他行的数组读入row中，并将row中最后一列提取
#出来放在labels中作为标签，并使用pop将该列从row去去除掉，最后将剩下的属性值转化为float类型存入xList中
xlist=[]
labels=[]
names=[]
firstline=True
for line in data:
    if firstline:
        names=line.strip().split(b';')
        firstline=False
    else:
        row=line.strip().split(b';')
        labels.append(float(row[-1]))
        row.pop()
        floatrow=[float(num) for num in row]
        xlist.append(floatrow)

#计算几行几列
nrows=len(xlist)
ncols=len(xlist[1])

#转化为numpy格式
x=numpy.array(xlist)
y=numpy.array(labels)
winenames=numpy.array(names)

#随机抽30%的数据用于测试，随机种子为531固定值，确保多次运行结果相同便于优化算法
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,random_state=531)

#初始化训练梯度模型的各种参数(根据效果微调)
#树的个数
nest=2000
#树的深度
depth=7
#梯度下降系数
learnrate=0.01
#0.5混杂随机选择属性，与随机森林算法结合
subsamp=0.5

#训练随机森林与梯度提升算法混杂的模型
winerandomforestmodel=ensemble.GradientBoostingRegressor(n_estimators=nest,max_depth=depth,learning_rate=learnrate,subsample=subsamp,loss='ls')
winerandomforestmodel.fit(xtrain,ytrain)

#迭代预测，方差加入mserror列表
mserror=[]
predictions=winerandomforestmodel.staged_predict(xtest)
for p in predictions:
    mserror.append(mean_squared_error(ytest,p))

print("MSE")
print(min(mserror))
print(mserror.index(min(mserror)))

#plot.figure:绘制多个图像
plot.figure()
plot.plot(range(1,nest+1),winerandomforestmodel.train_score_,label='training set mse')
plot.plot(range(1,nest+1),mserror,label='test set mse')
#图列
plot.legend(loc='upper right')
plot.xlabel('number of trees in ensemble')
plot.ylabel('mean squared error')
plot.show()
