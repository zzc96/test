'''
Created on Nov 22, 2010

@author: Peter
'''
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plotDrawPoint(plotPoint,bias,W1):
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    fr = open('testSet.txt')                              #打开测试集的文档
    for line in fr.readlines():                           #逐行读取数据
        lineSplit = line.strip().split('\t')              #去除头尾空格，并且以'\t'分割数据
        xPt = float(lineSplit[0])                         #获取第一维特征
        yPt = float(lineSplit[1])                         #获取第二维特征
        label = int(lineSplit[2])                         #获取标签
        if (label == -1):                                 #分类
            xcord0.append(xPt)
            ycord0.append(yPt)
        else:                                             #分类
            xcord1.append(xPt)
            ycord1.append(yPt)

    fr.close()                                            #关闭文件
    fig = plt.figure()                                    #生成绘画实例
    ax = fig.add_subplot(111)                             #生成画布，1行 1列 第一块
    ax.scatter(xcord0,ycord0, marker='s', s=25)           #画第一类标签点  s为大小
    ax.scatter(xcord1,ycord1, marker='o', s=25, c='red')  #画第二类标签点，c为颜色
    plt.title('Support Vectors Circled')                  #画标题
    for i in range(len(plotPoint)):                       #使用圆圈将支持向量标定出来
        circle = Circle((plotPoint[i][0], plotPoint[i][1]), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
        ax.add_patch(circle)

    #plt.plot([2.3,8.5], [-6,6]) #seperating hyperplane
    b = bias
    print("W1:",W1)
    w0 = W1[0][0]
    w1 = W1[1][0]
    x = arange(-2.0, 12.0, 0.1)
    y = (-w0*x - b)/w1                                     #0=w0x0+w1x1+w2x2 (x0=1) 此处x=x1；y=x2
    ax.plot(x,y)                                           #画拟合后的线
    ax.axis([-2,12,-8,6])                                  #画X轴和Y轴
    plt.show()

if __name__ == "__main__":
    plotDrawPoint()