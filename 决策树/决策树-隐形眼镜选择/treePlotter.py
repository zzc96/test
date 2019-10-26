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


def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]        #在父子节点间填充文本信息
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]

    createPlot.axl.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt):
    numLeafs = getNumLeafs(myTree)                                   #计算树的宽度
    depth = getTreeDepth(myTree)                                     #计算树的高度
    firstStr = list(myTree.keys())[0]                                #获得第一个名称
    #计算中间位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)                             #计算父节点和子节点的中间位置，并且添加文本标签信息
    plotNode(firstStr,cntrPt,parentPt,decisionNode)                  #绘制节点
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD              #按比例减少plotTree.yOff，并且标注此处要绘制子节点，应为我们是自顶向下绘制图形，所以需要依次递减y坐标值
    for key in secondDict.keys():                      #如果节点不是叶子节点，则递归，否则画出整个节点
        if type(secondDict[key]).__name__=="dict":
            plotTree(secondDict[key],cntrPt,str(key))                #递归整棵树
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW      #如果节点是叶子节点，则画出叶子节点
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD             #绘制完所有节点之后，增加Y的偏移


#plotTree.xOff和plotTree.yOff追踪已经绘制的节点位置，以及放置下一个节点的恰当位置
def createPlot(inTree):
    fig = plt.figure(1,facecolor="white")                    #创建一个图像实例，图像编号为1，背景色为白色
    fig.clf()                                                #清空绘图区
    axprops = dict(xticks=[],yticks=[])
    createPlot.axl = plt.subplot(111,frameon = False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))             #计算树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))            #计算树的深度
    plotTree.xOff = -0.5/plotTree.totalW                     #通过深度和宽度这两个信息计算树的摆放位置
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')                            #计算图信息并且绘制
    plt.show()                                               #显示

#获取叶节点的数目，所有子节点的数目
#第一个关键字是第一次划分数据集的类别标签。附带的数值表示子节点的取值
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]                              #获得第一个key
    secondDict = myTree[firstStr]                                  #获得第一个key下的所有内容
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":               #判定节点的数据类似是否为字典
            numLeafs += getNumLeafs(secondDict[key])               #递归  寻找叶子节点
        else:
            numLeafs += 1                                          #节点数目加1
    return numLeafs                                                #返回节点数目


#获得树的层数
#第一个关键字是第一次划分数据集的类别标签
#计算遍历过程中遇到的判断节点的个数，该函数的终止条件是叶子节点，一旦达到叶子节点，则从递归中返回
#并将计算树深度加一
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]                              #获得第一个key
    secondDict = myTree[firstStr]                                  #获得第一个key下的所有内容
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":               #判定节点的数据类似是否为字典
            thisDepth = 1 + getTreeDepth(secondDict[key])          #递归，层数+1
        else:
            thisDepth = 1                                          #不是字典的情况下，层数为1

        if thisDepth > maxDepth:                                   #寻找最大层数
            maxDepth = thisDepth

    return maxDepth                                                #返回最大层数


#预先存储树的信息，避免创建的麻烦
def retrieveTree(i):
    listOfTrees = [{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:"yes"}}}},
                   {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:"no"}}}}
                   ]
    return listOfTrees[i]

if __name__ == "__main__":
    # createPlot()
    myTree = retrieveTree(0)
    numLeafs = getNumLeafs(myTree)
    treeDepth = getTreeDepth(myTree)
    print(numLeafs,treeDepth)
    createPlot(myTree)
    myTree["no surfacing"][3] = 'maybe'
    createPlot(myTree)