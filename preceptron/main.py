from os import listdir
import numpy as np


def readData(fileFolder):
    fileList = listdir(fileFolder)
    trainDataSet= []
    trainLabel = []
    for fileName in fileList:
        #get the label
        label =[0 for i in range(10)]
        labelDigit = int(fileName.split('_')[0])
        label[labelDigit] = 1
        trainLabel.append(label)
        #get the data
        ifile = open(fileFolder+"/"+fileName)
        lines = ifile.readlines()
        dataSet = []
        for line in lines:
            line = line.split('\n')[0]
            for i in range(len(line)):
                dataSet.append(int(line[i]))
        trainDataSet.append(dataSet)

    return trainDataSet,trainLabel

# if (label == 0) ==> y=1 else y =-1
def changeFrom(trainLabel, index):
    recY = []
    for label in trainLabel:
        y = [-1]
        if (label[index] == 1):
            y[0] = 1
        recY.append(y)
    return recY


#train
def train_perceptron(trainDataset, y):
    m = len(trainDataset)
    n = len(trainDataset[0])
    learn_rate = 1
    # initialize
    w = np.zeros((1,n))
    b = 0
    #begin
    hasErrorData = True
    while(hasErrorData == True):
        hasErrorData = False
        for i in range(m):
            data = np.array(trainDataset[i])
            if (((w.dot(data.T) + b)*y[i][0] <= 0)):
                hasErrorData = True
                w = w + learn_rate * y[i][0] * data 
                b = b + learn_rate * y[i][0]
    return w,b


#def printAccuracyDigit(data):


def test_perceptron(testDataSet, testY, w, b):
    m = len(testDataSet)
    accuracyCount = 0
    for i in range(m):
        data = np.array(testDataSet[i])
        if((w.dot(data.T) + b) * testY[i][0] > 0):
            accuracyCount += 1
    print("Preceptron")
    print("test total number: %d , accuracy number: %d" % (m,int(accuracyCount)))
    return float(accuracyCount) / m





def main():
    #mnist
    
    trainDataSet, trainLabel = readData("data/trainingDigits")
    y = changeFrom(trainLabel, 0)    # 数字0作为正(y = 1)， 其他数字作为负（y = -1) 
    
    w, b = train_perceptron(trainDataSet, y)

    testDataSet, testLabel = readData("data/testDigits")
    testY = changeFrom(testLabel, 0)
    accuracyRate = test_perceptron(testDataSet, testY, w, b)
    print("accuracyRate: %f" % accuracyRate)
if __name__ == "__main__":
    main()