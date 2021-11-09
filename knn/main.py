import numpy as np
from os import listdir

def img2vector(filename):
    retrunVector = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            retrunVector[0,32 * i + j] = int(line[j])
    return retrunVector

def knnCore(trainMat, trainLabel, testData, k):
    m = trainMat.shape[0]

    diffMat = np.tile(testData,(m,1)) - trainMat
    squareDiffmat = diffMat ** 2
    squareDistance = squareDiffmat.sum(axis = 1)
    distances = squareDistance **0.5

    sortedDistance = distances.argsort()

    classCount = {}
    for i in range(k):
        votelLabel = trainLabel[sortedDistance[i]]
        classCount[votelLabel] = classCount.get(votelLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),key =lambda asd:asd[1],reverse=True)
    return sortedClassCount[0][0]

def main():
    trainLabel = []
    trainFileList = listdir('data/trainingDigits')
    m = len(trainFileList)
    trainMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainFileList[i]
        fileStr = fileNameStr.split('.')[0]
        label = int(fileStr.split('_')[0])
        trainLabel.append(label)
        trainMat[i,:] = img2vector('data/trainingDigits/%s' % fileNameStr)

    k = 3
    testFilelist = listdir('data/testDigits')
    accuracyCount = 0.0
    mTest = len(testFilelist)
    for i in range (mTest):
        fileNameStr = testFilelist[i]
        fileStr = fileNameStr.split('.')[0]
        testLabel = int(fileStr.split('_')[0])
        testData = img2vector('data/testDigits/%s' % fileNameStr)
        
        knnResult = knnCore(trainMat,trainLabel,testData,k)
        if (knnResult == testLabel):
            accuracyCount += 1
    print("KNN")
    print("test total number: %d , accuracy number: %d" % (mTest,int(accuracyCount)))
    print("accuracy Rate: %f" %(accuracyCount / float(mTest))) 
        
if __name__=="__main__":
    main()