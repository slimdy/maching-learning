import DecisionTree as DT
from pylab import mpl
mpl.rcParams.update({
    'font.family':'sans-serif',
    'font.sans-serif':['SimHei'],
})

"""
{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}}
"""
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    print(firstStr)
    secondDict = inputTree[firstStr]
    print(secondDict)
    featIndex = featLabels.index(firstStr)
    print(featIndex)
    classLabel = None
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
def grabTree(filename):
    import pickle
    pickle.load(open(filename))

"""
使用决策树预测隐形眼镜类型
tearRate 流泪率
pre 小孩
presbyopic 老花眼，远视的人 老人
myope  近视
hyperopia 远视
"""
def classifierOfLenses():
    fr  = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['年龄','病因','散光','流泪率']
    lensesTree = DT.createTree(lenses,lensesLabels)
    return lensesTree
if __name__ == '__main__':
    # myData,labels = DT.createDataSet()
    # inputTree = DT.retrieveTree(0)
    # classLabel = classify(inputTree,labels,[1,1])
    # print(classLabel)

    #run the classifier of lenses:
    lensesTree =  classifierOfLenses()
    DT.createPlot(lensesTree)