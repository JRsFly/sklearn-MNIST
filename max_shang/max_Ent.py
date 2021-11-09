import numpy as np
from os import listdir
def readData(fileFolder):
    fileList = listdir(fileFolder)
    trainDataSet = []
    trainLabel = []
    for filename in fileFolder:

        fileNameStr = filename.split('.')[0]
        label = int(fileNameStr.spilt('_')[0])
        trainLabel.append(label)

        ifile = open(fileFolder+'/'+filename)
        lines = ifile.readlines()
        dataSet = []
        for line in lines:
            line = line.split('\n')[0]
            for i in range(len(line)):
                dataSet.append(int(line[i]))
            trainDataSet.append(dataSet)
    return trainDataSet, trainLabel

def changeForm(trainLabel, index):
    recY = []
    y = 0
    for i in range(len(trainLabel)):
        if (trainLabel[i] == index):
            y = 1
        else:
            y = 0
        recY.append(y)
    return recY
def calExpy():
    Ep_xy = []
def maxEntropyTrain(trainDataSet, trainLabel, iteration = 500):
    for i in range(iteration):

        Expy = calcExpy()
def main():
    # 获取训练集及标签
    print('start read transSet')
    trainData, trainLabel = readData('data/trainingDigits')
    trainY = changeForm(trainLabel, 0)
    # 获取测试集及标签
    print('start read testSet')
    testData, testLabel = readData('data/testDigits')
    testY = changeForm(testLabel, 0)
    #初始化最大熵类
    maxEnt = maxEnt(trainData, trainY, testData, testY)

    #开始训练
    print('start to train')
    maxEnt.maxEntropyTrain()

    #开始测试
    print('start to test')
    accuracy = maxEnt.test()
    print('the accuracy is:', accuracy)
if __name__ == "__main__":
    main()