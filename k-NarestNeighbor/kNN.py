import numpy as np
import operator
import matplotlib.pyplot as plt
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ["A","A","B","B"]
    return group,labels
#k-近邻算法
def classify0(inX,dataSet,labels,k):
    #计算距离
    # print(dataSet.shape)
    dataSetSize = dataSet.shape[0] #4
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet#tile把inX 按照（4，1）的方式补全。4的意思是x轴上copy四次
    # print(diffMat)
    #这里进行了一个数学运算 (x*x +y*y)然后开方
    sqDiffMat = diffMat **2#2次方计算距离
    # print(sqDiffMat)
    # exit()
    sqDistances = sqDiffMat.sum(axis=1) #axis = 0 是按列相加(x+x y+y)，axis = 1是矩阵中每个向量自我相加（x+y）
    # print(sqDistances)
    distances = sqDistances ** 0.5 #开方
    # print(distances)
    sortedDistIndicies = np.argsort(distances) # 把距离参照点的距离按照升序排序
    # print(sortedDistIndicies)
    classCount = {}
    #选择最小距离的K个点
    for  i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1 #取之前的某一个分类，给他的个数加1
    #排序
    # print(classCount.items())
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)#把字典items的第1个域，也就是格式按照降序排序
    # print(sortedClassCount)
    return  sortedClassCount[0][0]
#测试简单的近邻算法
# groups,labels = createDataSet()
# print(groups)
# print(labels)
# print(classify0([0.9,0.9],groups,labels,3))

#近邻算法应用于婚介
def file2matrix(fileName):
    fr = open(fileName)
    arrayOLines = fr.readlines() #读取文件全部内容 并按行弄成列表
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]#每行修改数据为listfromLine的前三个
        if listFromLine[-1] == 'largeDoses':
            lastItem = 3
        elif listFromLine[-1] == 'smallDoses':
            lastItem = 2
        else:
            lastItem = 1
        classLabelVector.append(lastItem)
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normaDataSet = dataSet - np.tile(minVals,(m,1))
    normaDataSet = normaDataSet / np.tile(ranges, (m, 1))
    return  normaDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.80
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classiFierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with:%d,the real answer is: %d"%(classiFierResult,datingLabels[i]))
        if (classiFierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is : %f"%(errorCount/float(numTestVecs)))
# datingClassTest()
# DataMat,DataLabels = file2matrix('datingTestSet.txt')
# print(DataMat)
# print(autoNorm(DataMat))
# print(DataLabels)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(DataMat[:,1],DataMat[:,2],15.0*np.array(DataLabels),15.0*np.array(DataLabels))
# plt.show()
#婚介近邻函数的实现
def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input("frequent fliter miles earned per year"))
    iceCream = float(input("Liters of ice cream consumed per year"))
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("you will probably like this person:",resultList[classifierResult -1])
classifyPerson()
#冰雹猜想
# def BB(n,count = 0,l=[]):
#     l.append(n)
#     if count != 0:
#         if n == 1 or n == 4 or n == 2:
#             print(n,count)
#             if n == 1:
#                 return n,count,l
#     if n % 2 == 1 :
#         return BB(3*n+1,count+1,l)
#     else:
#         return BB(n/2,count+1,l)
# def BB2(n,count = 0,l=[]):
#     l.append(n)
#     if count < 200:
#         if n % 2 == 1:
#             return BB2(3 * n + 1, count + 1,l)
#         else:
#             return BB2(n / 2, count + 1,l)
#     else:
#         return l
# n,count,l = BB(27)
# l = BB2(27)
# print(n)
# print(count)
# print(l)