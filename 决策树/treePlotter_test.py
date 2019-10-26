import matplotlib.pyplot as plt

#定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth",fc="0.8")   #波浪
leafNode = dict(boxstyle="round4",fc="0.8")         #弧线
arrow_args = dict(arrowstyle="<-")

#执行实际绘图功能
#节点文字，坐标，父节点坐标，节点类型
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.axl.annotate(nodeTxt,xy=parentPt,xycoords="axes fraction",xytext=centerPt,textcoords="axes fraction",\
                            va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1,facecolor='white')                    #创建一个图像实例，图像编号为1，背景色为白色
    fig.clf()                                                #清空绘图区
    #定义绘图区
    createPlot.axl = plt.subplot(111,frameon=False)          #创建单个子图，行数为111
    plotNode("DecisionNode",(0.5,0.1),(0.1,0.5),decisionNode)#画从（0.5，0.1）到（0.1，0.5的箭头线），并框起来显示文字
    plotNode('LeafNode',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()                                               #显示

if __name__ == "__main__":
    createPlot()