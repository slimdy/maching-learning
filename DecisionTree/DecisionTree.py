from math import log
from pylab import mpl
import operator
import matplotlib.pyplot as plt

"""
什么是香农熵
熵是起初是热力学概念，它的物理意义是体系混乱程度的变量
熵越大，混乱程度越大
香农熵是由香农在1945年提出的如何解决信息的量化度量问题，一条信息的信息量大小和他的不确定性有直接的关系。例如如果我们要知道一个非常不确定的事情，
或者是我们一无所知的事情。就需要大量的信息，反之，当这个事情我们已经相当了解了。则不需要多少信息。从这个角度，我们可以认为信息的量度就等于不确定性的多少

香农熵的单位是bit  2的次方，所以公式里有log2.举个例子，世界杯32支球队，如果在不知道谁夺冠的情况下，想要知道那支队伍得到冠军，在对方不告诉我们的前提下
最少要花5次的提问才能，1-16 17-32    1-8 9-16    1-4 5-8     1-2 2-4     1  2 通过这5次的提问我们就知道了谁是冠军。这个五次就是log2 32 = 5
信息量就是5.但是其实我们已经掌握了更加所得信息，有人就是每个队伍的夺冠的概率并不一样，我们可以把夺冠热门分为一组，把其他队伍分为一组。这样我们字需要在夺冠
热门队伍里面寻找就可以了。

香农认为，获得信息，5次（32各球队举例子）是最大的次数 或者更少，所以它用一个人公式表示了这种变化
香农熵公式
-（p1*log p1+p2*log p2+p3*log p3+p4*logp4+...+p32*log p32）
也就是 -∑p*log p

"""
#计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] +=1
    shannonEnt = 0.0
    # print(labelCounts)
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        # print(prob)
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels
#拆分原始数据集 参数：待划分数据集，划分特征，需要返回特征的值
"""
[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
循环整个dataSet，将它们的每个元素第axis项和value作对比，如果相等，说明是属于value 这个分组的
例如：axis = 0 value = 1 就意思是安装每个元素的第一项的值如果等于 value 也就是等于1 ，则属于1这个分组的 [1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no']
     而[0, 1, 'no'], [0, 1, 'no']就是另一组的
     axis = 1,value = 1 就是 [1, 1, 'yes'], [1, 1, 'yes'],  [0, 1, 'no']，[0, 1, 'no'] 而[1, 0, 'no'] 是单另一组的
"""
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

"""
此函数是通过香农熵来判断，判断哪个数据分类更好
原理：熵越大，分类越混乱，如果一种分类方法的熵值小于其他的分类方法，则说明第一种好。
过程：[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
1.计算整个dataSet的熵值0.9左右
2.设置最有的熵值 也就是0，和最优的数组元素索引 -1的初始值
3.循环dataSet每个元素中的前两个，将每一次循环所得到的数据值放在一个新的数组，分别是[1,1,1,0,0] [1,1,0,1,1]
4.将它变成不可重复的数据集合set，然后再用这其中的两个不同的值分别拆分数据集。
5.计算每个数据集的香农熵，并和完美的作比较，去最接近0的那个索引，作为最有分类索引
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    # print(numFeatures)
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # print(featList)
        # print(uniqueVals)
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            # print(subDataSet)
            # print(newEntropy)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
"""
用例子来解释：
该函数传递2个参数：dataSet 是数据集，label 类别名
dataSet：[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
label:[no surfacing,'flippers']
1.首先是将dataSet每一项的最后一个元素['yes','yes','no','no','no']
2.判断一下进来的dataSet每一项最后一个元素是否只剩一下，如果只剩一个，则表明是已经分类完成，直接返回
3.在判断一下判断的条件是否没有了，如果没有了则采取多数投票的方式判断类别直接返回
4.如果都不符合上述条件，则开始分类，使用chooseBestFeatureToSplit方法寻找最优的分类索引 也就是0
5.然后在dataSet 中的每个元素里提取安装最有分类索引所得到的数据，组成一个set 也就是[[1,1,'yes']] 取它的第0项也就是第一个 1.取完则得(1,1,1,0,0) set(1,0)
6.接着循环整个set(0,1) ,通过splitDataSet函数将dataSet按照最优分类索引 分别对set（0，1） 进行分类，分出来的[1,'no'][1,'no']和[1,yes][1,yes][0m'no']
7.让后将这两组分别当做是dataset 迭代使用createTree 继续分类。第一组 也就是[1,'no'][1,'no'] 没有其他分类(只有no) 所以他们的分类就是no，第二组因为包含yes和no所以继续循环
8.最后分出 {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
"""
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    # print(classList)
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    print(bestFeat)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
#一下都是绘图
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    arrow_args = dict(arrowstyle='<-')
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)
def createPlot(inTree):
    decisionNode = dict(boxstyle='sawtooth',fc='0.8')
    leafNode = dict(boxstyle='round4',fc='0.8')
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    # plotNode(plotNode'决策节点',(0.5,0.1),(0.1,0.5),decisionNode)
    # plotNode('叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs
def getTreeDepth(myTree):
    maxDepth=0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth
def retrieveTree(i):
    listOfTrees = [{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},{'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
    return listOfTrees[i]
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1]) / 2.0 + cntrPt[1]
    # createPlot().ax1.text(xMid,yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt):
    decisionNode = dict(boxstyle='sawtooth', fc='0.8')
    leafNode = dict(boxstyle='round4', fc='0.8')
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff+(1.0+float(numLeafs)) /2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

if __name__ == '__main__':
    dataSet , labels = createDataSet()
    # print(createTree(dataSet,labels))
    # createPlot()
    myTree = retrieveTree(0)
    createPlot(myTree)

