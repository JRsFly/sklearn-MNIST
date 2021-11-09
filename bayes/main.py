from os import listdir
import numpy as np
from math import log

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
def Cal_label(trainLabel,lambd):
    label_count = {}
    for i in range(len(trainLabel)):
        votelLabel = trainLabel[i]
        label_count[votelLabel] = label_count.get(votelLabel, 0) + 1
    for i in range(len(label_count)):
        label_count[i] += lambd
    return label_count


def train_bayes(trainDataSet,trainLabel,labelCount,lambd):
    m = len(trainDataSet)
    x_count = []
    P_x_y = np.zeros((len(labelCount),len(trainDataSet[0])))
    for i in range(m):
        data = np.array(trainDataSet[i])
        P_x_y[trainLabel[i],:] = P_x_y[trainLabel[i],:] + data
    P_x_y = P_x_y / 1.0
    for i in range(len(labelCount)):
        icount = labelCount[i]
        P_x_y[i,:] = (P_x_y[i,:] + lambd) / (icount + 2 * lambd)
    return P_x_y
def test_bayes(testDataSet,testLabel,P_x_y,labelCount):
    m = len(testDataSet)
    accuracycount = 0
    for mtest in range(m):
        
        bayesResult = 0
        bayesResult_max = log(labelCount[bayesResult])
        for i in range(len(P_x_y[0,:])):
            if (testDataSet[mtest][i] == 1):
                bayesResult_max += log(P_x_y[bayesResult][i])
            else:
                bayesResult_max +=  log(1 - P_x_y[bayesResult][i])

        for j in range(len(P_x_y[:,0])):
            j_max = log(labelCount[j])
            for i in range(len(P_x_y[0,:])):
                if (testDataSet[mtest][i] == 1):
                    j_max += log(P_x_y[j][i])
                else:
                    j_max += log(1 - P_x_y[j][i])
            if (j_max > bayesResult_max):
                bayesResult_max = j_max
                bayesResult = j
        if (bayesResult == testLabel[mtest]):
            accuracycount += 1
        print("accuarcy result:%d  bayes result:%d" % (testLabel[mtest],bayesResult))
    return accuracycount
        
def main():
    #mnist
    lambd = 1 #拉普拉斯平滑系数
    trainDataSet, trainLabel = readData("data/trainingDigits")
    labelCount = Cal_label(trainLabel,lambd)
    
    P_x_y = train_bayes(trainDataSet,trainLabel,labelCount,lambd)
    testDataSet, testLabel = readData("data/testDigits")
    accuracyCount = test_bayes(testDataSet,testLabel,P_x_y,labelCount)
    print("accuracyRate: %f" % (float(accuracyCount) / len(testDataSet)))
if __name__=="__main__":
    main()