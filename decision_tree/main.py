from os import listdir
import numpy as np
from numpy.core.defchararray import index
from numpy.core.fromnumeric import _cumprod_dispatcher

#read the data
def readDataSet(fileDir):
    fileList = listdir(fileDir)
    m = len(fileList)
    dataSet = []
    index = 0
    for filename in fileList:
        fr = open(fileDir + "/" + filename)
        data = []
        for i in range(32):
            line = fr.readline()
            for j in range(32):
                data.append(int(line[j]))
        dataSet.append(data)
        dataLabel = int((filename.split('.')[0]).split('_')[0])
        dataSet[index][-1] = str(dataLabel)
        index += 1
    featureNameSet = []
    #获取每个向量的特征数目，去掉重复项并保存
    for i in range(len(dataSet[0]) - 1):
        col = [x[i] for x in dataSet]
        colSet = set(col)
        featureNameSet.append(list(colSet))
    return dataSet, featureNameSet

def calcEntropy(dataSet):
    mD = len(dataSet)
    dataLabelList = [x[-1] for x in dataSet]
    dataLabelSet = set(dataLabelList)
    ent = 0
    for label in dataLabelList:
        mDv = dataLabelList.count(label)
        prop = float(mDv) / mD
        ent = ent - prop * np.math.log(prop, 2)
    
    return ent 

#   拆分数据集
#   index ---   拆分特征下标
#   feature --- 要拆分的特征
#   返回值 ---  dataSet中index所在特征为feature，且去掉index一列的集合
def splitDataSet(dataset, index, feature):
    splitDataSet = []
    for data in dataset:
        if (data[index] == feature):
            reduceFeatureVec = data[:index]
            reduceFeatureVec.extend(data[index + 1:])
            splitDataSet.append(reduceFeatureVec)
    return splitDataSet

#根据信息增益 -- 选择最优特征
#返回值 - 最优特征下标
def chooseBestFeature(DataSet):
    entD = calcEntropy(DataSet)
    mD = len(DataSet)
    featureNumber = len(DataSet[0]) - 1
    maxGain = -100
    maxIndex = -1
    for i in range(featureNumber):
        entDCopy = entD
        featureI = [x[i] for x in DataSet]
        featureSet = set(featureI)
        for feature in featureSet:
            splitedDataSet = splitDataSet(DataSet, i, feature)
            mDv = len(splitedDataSet)
            entDCopy = entDCopy - float(mDv) / mD * calcEntropy(splitedDataSet)
        if (maxIndex == -1):
            maxGain = entDCopy
            maxIndex = i
        elif(maxGain < entDCopy):
            maxGain = entDCopy
            maxIndex = i
    return maxIndex
#寻找最多的，作为标签
def mainLabel(labelList):
    labelRec = labelList[0]
    maxLabelCount = -1
    labelSet = set(labelList)
    for label in labelSet:
        if (labelList.count(label) > maxLabelCount):
            maxLabelCount = labelList.count(label)
            labelRec = label
    return labelRec
def createFullDecisionTree(dataSet, featureNames, featureNameSet, labelListParent):
    labelList = [x[-1] for x in dataSet]
    if (len(dataSet) == 0):
        return mainLabel(labelListParent)
    elif (len(dataSet[0]) == 1):
        return mainLabel(labelList)
    elif(labelList.count(labelList[0]) == len(labelList)):
        return labelList[0]
    
    bestFeatureIndex = chooseBestFeature(dataSet)
    #print(bestFeatureIndex)
    bestFeatureName = featureNames.pop(bestFeatureIndex)
    myTree = {bestFeatureName : {}}
    featureList = featureNameSet.pop(bestFeatureIndex)
    
    ##dfassd
    featureSet = set(featureList)
    for feature in featureSet:
        featureNamesNext = featureNames[:]
        featureNameSetNext = featureNameSet[:][:]
        splitedDataSet = splitDataSet(dataSet, bestFeatureIndex, feature)
        myTree[bestFeatureName][feature] = createFullDecisionTree(splitedDataSet, featureNamesNext, featureNameSetNext, labelList)
    return myTree

def saveMytree(mytree, filename):
    s = str(mytree)
    ofile = open(filename,'w')
    ofile.writelines(s)
    ofile.close()

def loadMyTree(fileName):
    ifile = open(fileName,'r')
    s = ifile.readline()
    mytree = eval(s)
    return mytree

def getlabel(myTree, data):
    keys = myTree.keys()
    for key in keys:
        subTree = myTree[key]
        index = int(key)
        feature = int(data[index])
        nextTree = subTree[feature]
        if (type(nextTree).__name__ == 'dict'):
            classLabel = getlabel(nextTree,data)
        else:
            classLabel = int(nextTree)
    return classLabel
def test(myTree, testDataSet):
    errorCount = 0
    m = len(testDataSet)
    for data in testDataSet:
        realLabel = int(data[-1])
        testLabel = getlabel(myTree, data[:-1])
        print("The right label:%d   The decsion_tree label: %d" % (realLabel,testLabel))
        if (testLabel != realLabel):
            errorCount += 1
    print("Accuracyrate: %f" %  (1 - (float(errorCount) / m)))
def main():
    trainingDataSet, featureNameSet = readDataSet("data/trainingDigits")
    featureNames = [str(x) for x in range(1024)]
    myTree = createFullDecisionTree(trainingDataSet,featureNames,featureNameSet, [])
    saveMytree(myTree,"myTree.txt")

    myTree = loadMyTree("myTree.txt")
    testDataSet, featureNameSet = readDataSet("data/testDigits")
    test(myTree, testDataSet)

if __name__ == "__main__":
    main()
