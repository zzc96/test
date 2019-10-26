from numpy import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plotRBF(plotPoint):
    xcord0 = []
    ycord0 = []
    xcord1 = []
    ycord1 = []
    fr = open('./testSetRBF.txt')  # 打开测试集的文档
    for line in fr.readlines():  # 逐行读取数据
        lineSplit = line.strip().split('\t')  # 去除头尾空格，并且以'\t'分割数据
        xPt = float(lineSplit[0])  # 获取第一维特征
        yPt = float(lineSplit[1])  # 获取第二维特征
        label = int(float(lineSplit[2]))  # 获取标签
        if (label == -1):  # 分类
            xcord0.append(xPt)
            ycord0.append(yPt)
        else:  # 分类
            xcord1.append(xPt)
            ycord1.append(yPt)

    fr.close()  # 关闭文件
    fig = plt.figure()  # 生成绘画实例
    ax = fig.add_subplot(111)  # 生成画布，1行 1列 第一块
    ax.scatter(xcord0, ycord0, marker='s', s=25)  # 画第一类标签点  s为大小
    ax.scatter(xcord1, ycord1, marker='o', s=25, c='red')  # 画第二类标签点，c为颜色
    plt.title('Support Vectors Circled')  # 画标题

    for i in range(len(plotPoint)):                       #使用圆圈将支持向量标定出来
        circle = Circle((plotPoint[i][0], plotPoint[i][1]), 0.05, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
        ax.add_patch(circle)

    plt.show()

if __name__ == "__main__":
    plotRBF()
