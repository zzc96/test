# coding:utf-8
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy

#定义一些简单数据，用于计算
def load_data():
    dataSet = [
        [150, 6450],                                                    #房子面积、单价
        [200, 7450],
        [250, 8450],
        [300, 9450],
        [350, 11450],
        [400, 15450],
        [600, 18450]
    ]

    X_parameter = []                                                    #初始化数据集X
    Y_parameter = []                                                    #初始化数据集Y
    for x_y in dataSet:                                                 #轮询数据
        X_parameter.append(float(x_y[0]))                               #取出房子面积数据
        Y_parameter.append(float(x_y[1]))                               #取出房子单价数据

    return X_parameter, Y_parameter                                     #返回数据


# 线性回归分析
#X_parameter   房子面积值
#Y_parameter   房子单价值
def linear_model_main(X_parameter, Y_parameter):
    X_parameter = numpy.mat(X_parameter).T                               #将列表转换成矩阵，并且进行转置
    Y_parameter = numpy.mat(Y_parameter).T

    regr = LinearRegression()
    regr.fit(X_parameter, Y_parameter)

    # 3. 构造返回字典
    lineParame = {}
    # 3.1 截距值
    lineParame['intercept'] = regr.intercept_
    # 3.2 回归系数（斜率值）
    lineParame['coefficient'] = regr.coef_

    return lineParame

def predictHouseValue(lineParame,preValue):
    return lineParame['intercept'][0] + lineParame['coefficient'][0][0] * preValue

# 绘出图像
def show_linear_line(lineParame,X,Y):
   fig = plt.figure()
   ax = fig.add_subplot(111)

   x = numpy.mat(range(100,700)).T

   ax.scatter(X,Y,c="red")

   ax.plot(x,lineParame['intercept'][0] + lineParame['coefficient'][0][0] * x)

   plt.title("Line")
   plt.xlabel("House Size")
   plt.ylabel("House Prize")

   plt.show()

if __name__ == '__main__':
    # main()
    X,Y = load_data()
    predict_square_feet = 700
    result = linear_model_main(X, Y)
    for key, value in result.items():
        print('{0}:{1}'.format(key, value))

    preD = predictHouseValue(result,700)
    print(preD)
    show_linear_line(result,X,Y)