import numpy as np
from os import listdir
import operator
def img2vector(fileName):
    returnVect = np.zeros((1,1024))
    fr = open(fileName)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
testVector = img2vector('testDigits/0_13.txt')
# print(testVector[0,32:63])
#k-近邻算法
"""
例如 
如果样本是 [[0,0.1],[0.1,0.1],[1,1.2],[0.8,1]] 
分类名称是 靠近0的称为A，靠近1的我们称为B
识别的是[0.2,0]
1.首先将[0.2,0] 变成和样本数据一样大小 也就是在x轴上赋值4次变成[[0.2,0],[0.2,0],[0.2,0],[0.2,0]]
2.用样本矩阵减去数据矩阵，平方。变成[[0.64,0.01],[0.01,0.01],[0.64,1.44],[0.36,1]]
3.将平方的结果 按照每个向量自我相加 [0.65，0.02，2.08，1.36]，加完后在一个个的开方，产生距离数组 [0.80622,0.141421,1.44222,1.16619]
4.把开方的结果 升序排序[0.141421 0.80622,1.16619,1.44222] 并且得到索引数组[1,0,3,2]
5.根据K值 来选取索引排序的结构，例如K等于3  也就是说选取1，0，3 三个所以 他们和分类名称[A，A，B，B] 也就是说 A A B 被选择出来了
6.分析结果 将A出现的次数和B出现的次数做降序排序，A>B
7.返回A 也就是说A 是最终结果

k值的选择
K 值的选择会对算法的结果产生重大影响。K值较小意味着只有与输入实例较近的训练实例才会对预测结果起作用，
但容易发生过拟合；如果 K 值较大，优点是可以减少学习的估计误差，但缺点是学习的近似误差增大，这时与输入实例较远的训练实例也会对预测起作用，使预测发生错误。
在实际应用中，K 值一般选择一个较小的数值，通常采用交叉验证的方法来选择最优的 K 值。
"""
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

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    print(trainingFileList)
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNameStr)
    print(hwLabels)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s'% fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print('the classifier came back with :%d,the real answer is:%d' %(classifierResult,classNumStr))
        if classifierResult != classNumStr:
            # command = input('occur error，continue？')
            # if int(command) != 1:
            #     exit()
            errorCount += 1
    print('\nthe total number of errors is :%d' % errorCount)
    print('\nthe total error rate is : %f'%(errorCount/float(mTest)))
handwritingClassTest()