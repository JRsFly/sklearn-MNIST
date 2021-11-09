from os import listdir
import numpy as np
from matplotlib import pyplot as plt


def readData(fileFolder):
    fileList = listdir(fileFolder)
    trainDataSet = []
    trainLabel = []
    for filename in fileList:
        
        #get the label
        fileNameStr = filename.split('.')[0]
        label = int(fileNameStr.split('_')[0])
        trainLabel.append(label)
        #get the data
        ifile = open(fileFolder+"/"+filename)
        lines = ifile.readlines()
        dataSet = []
        for line in lines:
            line = line.split('\n')[0]
            for i in range(len(line)):
                dataSet.append(int(line[i]))
        trainDataSet.append(dataSet)
    return trainDataSet,trainLabel

# if (label == index) ==> y=1 else y =-1
def changeFrom(trainLabel, index):
    recY = []
    y = 0
    for i in range(len(trainLabel)):
        if (trainLabel[i] == index):
            y = 1
        else: 
            y = 0
        recY.append(y)
    return recY


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#训练
#iteration默认为100，即100次迭代
def train_logistics(trainDataSet, trainLabel, iteration = 100):
    #转化为矩阵
    trainDataMat = np.mat(trainDataSet)
    #trainLabelMat = np.mat(trainLabel).transpose()
    m, n = np.shape(trainDataMat)
    w = np.ones((n, 1))

    #开始迭代
    for i in range(iteration):
        indexList = list(range(m))
        for j in range(m):
            alpha = 4 / (1.0 + i + j) * 0.01  #学习速率动态变化
            index = int(np.random.uniform(0, len(indexList)))  #从indexList中取出一个index用于训练
            z = sigmoid(np.sum(trainDataMat[index] * w))
            dw = (trainLabel[index] - z) * trainDataMat[index].transpose()
            w = w + alpha * dw
            del(indexList[index])
    return w

def test_logistics(testDataSet, testLabel, w):
    testDataMat = np.mat(testDataSet)
    m, n = np.shape(testDataMat)
    rightCount = 0
    for i in range(m):
        z = sigmoid(np.sum(testDataMat[i] * w))
        if(z > 0.5 and testLabel[i] == 1 or z < 0.5 and testLabel[i] == 0):
            rightCount += 1
            print("right! you Label %f" % z)
        else:
            print("error! real Label: %d, you Label %f" % (testLabel[i], z))
    return float(rightCount) / m





def main():
    #mnist
    
    trainDataSet, trainLabel = readData("data/trainingDigits")
    y = changeFrom(trainLabel, 0)    # 数字0作为正(y = 1)， 其他数字作为负（y = -1) 
    w= train_logistics(trainDataSet, y, iteration= 500)

    testDataSet, testLabel = readData("data/testDigits")
    testY = changeFrom(testLabel, 0)
    accuracyRate = test_logistics(testDataSet, testY, w)
    print("total: %d, accuracyRate: %f" % (len(testDataSet), accuracyRate))
if __name__ == "__main__":
    main()