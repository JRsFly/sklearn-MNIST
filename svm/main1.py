from sys import dont_write_bytecode
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from numpy.core.defchararray import _just_dispatcher, index
from numpy.lib import index_tricks
from numpy.lib.function_base import select
def changeFrom(trainLabel, index = 1):
    recY = []
    y = 1
    for i in range(len(trainLabel)):
        if (trainLabel[i] == index):
            y = 1
        else: 
            y = -1
        recY.append(y)
    return recY

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
    return np.array(trainDataSet),np.array(changeFrom(trainLabel))
class SVM:
    def __init__(self, max_iter= 500, kneral="linear"):
        self.max_iter = max_iter
        self._kneral = kneral
    
    def init_args(self,features,labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0
        self.alpha = np.ones(self.m)
        self.computer_product_matrix()
        
        self.C = 1.0
        self.create_E()
    def judge_KKT(self,i):
        y_g = self.function_g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 <self.alpha[i] <self.C:
            return y_g == 1
        else:
            return y_g <= 1

    def computer_product_matrix(self):
        self.product_matrix = np.zeros((self.m,self.m)).astype(np.float)
        for i in range(self.m):
            for j in range(self.m):
                if self.product_matrix[i][j] == 0.0:
                    self.product_matrix[i][j] = self.product_matrix[j][i]=self.kernel(self.X[i],self.X[j])
    def kernel(self,x1,x2):
        if self._kneral == 'linear':
            return np.dot(x1,x2)
        elif self._kneral == 'poly':
            return (np.dot(x1,x2) + 1) ** 2
        return 0
    
    def create_E(self):
        self.E = (np.dot((self.alpha * self.Y),self.product_matrix) + self.b) - self.Y
    def function_g(self, i):
        return self.b + (np.dot((self.alpha * self.Y),self.product_matrix[i]))
    
    def select_alpha(self):

        index_list = [i for i in range(self.m) if 0 <self.alpha[i] < self.C]
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        for i in index_list:
            if self.judge_KKT(i):
                continue
            E1 = self.E[i]

            if E1 >=0 :
                j = np.argmin(self.E)
            else:
                j = np.argmax(self.E)
            return i,j
    def clip_alpha(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha
    def Train(self, features,labels):
        self.init_args(features,labels)

        for t in range(self.max_iter):
            i1, i2 = self.select_alpha()

            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            # eta=K11+K22-2K12
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2]) - 2 * self.kernel(
                self.X[i1], self.X[i2])
            if eta <= 0:
                # print('eta <= 0')
                continue

            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta  # 此处有修改，根据书上应该是E1 - E2，书上130-131页
            alpha2_new = self.clip_alpha(alpha2_new_unc, L, H)

            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - self.Y[
                i2] * self.kernel(self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - self.Y[
                i2] * self.kernel(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.create_E()#这里与书上不同，，我选择更新全部E

    def predict(self, data):
        r = self.b
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])

        return 1 if r > 0 else -1

    def score(self, X_test, y_test):
        right_count = 0
        for i in range(len(X_test)):
            result = self.predict(X_test[i])
            if result == y_test[i]:
                right_count += 1
        return right_count / len(X_test)
print(__name__)
def main():
    svm = SVM(max_iter=100)
    trainDataSet, trainLabel = readData('data/trainingDigits')
    testDataSet, testLabel = readData('data/testDigits')
    svm.Train(trainDataSet,trainLabel)
    print(1)
    print("The accurate rate:%f" % svm.score(testDataSet,testLabel))
if __name__ == "__main__":
    main()