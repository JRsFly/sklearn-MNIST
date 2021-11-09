import time
import math
import numpy as np
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    return trainDataSet, changeFrom(trainLabel)
sign_time_count = 0

class Sign(object):
    '''
    阈值分类器

    有两种方向，
        1）x<v y=1
        2) x>v y=1
        v 是阈值轴

    因为是针对已经二值化后的MNIST数据集，所以v的取值只有3个 {0,1,2}
    '''

    def __init__(self,features,labels,w):
        self.X = features               # 训练数据特征
        self.Y = labels                 # 训练数据的标签
        self.N = len(labels)            # 训练数据大小

        self.w = w                      # 训练数据权值分布

        self.indexes = [0,1,2]          # 阈值轴可选范围

    def _train_less_than_(self):
        '''
        寻找(x<v y=1)情况下的最优v
        '''

        index = -1
        error_score = 1000000

        for i in self.indexes:
            score = 0
            for j in range(self.N):
                val = -1
                if self.X[j]<i:
                    val = 1
                if val*self.Y[j]<0:
                    score += self.w[j]

            if score < error_score:
                index = i
                error_score = score

        return index,error_score



    def _train_more_than_(self):
        '''
        寻找(x>v y=1)情况下的最优v
        '''

        index = -1
        error_score = 1000000

        for i in self.indexes:
            score = 0
            for j in range(self.N):
                val = 1
                if self.X[j]<i:
                    val = -1

                if val*self.Y[j]<0:
                    score += self.w[j]

            if score < error_score:
                index = i
                error_score = score

        return index,error_score

    def train(self):
        global sign_time_count
        time1 = time.time()
        less_index,less_score = self._train_less_than_()
        more_index,more_score = self._train_more_than_()
        time2 = time.time()
        sign_time_count += time2-time1

        if less_score < more_score:
            self.is_less = True
            self.index = less_index
            return less_score

        else:
            self.is_less = False
            self.index = more_index
            return more_score

    def predict(self,feature):
        if self.is_less>0:
            if feature<self.index:
                return 1.0
            else:
                return -1.0
        else:
            if feature<self.index:
                return -1.0
            else:
                return 1.0


class AdaBoost(object):

    def __init__(self):
        pass

    def _init_parameters_(self,features,labels):
        self.X = features                           # 训练集特征
        self.Y = labels                             # 训练集标签

        self.n = len(features[0])                   # 特征维度
        self.N = len(features)                      # 训练集大小
        self.M = 10                                 # 分类器数目

        self.w = [1.0/self.N]*self.N                # 训练集的权值分布
        self.alpha = []                             # 分类器系数  公式8.2
        self.classifier = []                        # (维度，分类器)，针对当前维度的分类器

    def _w_(self,index,classifier,i):
        '''
        公式8.4不算Zm
        '''

        return self.w[i]*math.exp(-self.alpha[-1]*self.Y[i]*classifier.predict(self.X[i][index]))

    def _Z_(self,index,classifier):
        '''
        公式8.5
        '''

        Z = 0

        for i in range(self.N):
            Z += self._w_(index,classifier,i)

        return Z

    def train(self,features,labels):

        self._init_parameters_(features,labels)

        for times in range(self.M):

            best_classifier = (100000,None,None)        #(误差率,针对的特征，分类器)
            for i in range(self.n):
                features = list(map(lambda x:x[i],self.X))
                classifier = Sign(features,self.Y,self.w)
                error_score = classifier.train()

                if error_score < best_classifier[0]:
                    best_classifier = (error_score,i,classifier)

            em = best_classifier[0]

            

            if em==0:
                self.alpha.append(100)
            else:
                self.alpha.append(0.5*math.log((1-em)/em))

            self.classifier.append(best_classifier[1:])

            Z = self._Z_(best_classifier[1],best_classifier[2])

            # 计算训练集权值分布 8.4
            for i in range(self.N):
                self.w[i] = self._w_(best_classifier[1],best_classifier[2],i)/Z

    def _predict_(self,feature):

        result = 0.0
        for i in range(self.M):
            index = self.classifier[i][0]
            classifier = self.classifier[i][1]

            result += self.alpha[i]*classifier.predict(feature[index])

        if result > 0:
            return 1
        return -1



    def predict(self,features):
        results = []

        for feature in features:
            results.append(self._predict_(feature))

        return results


if __name__ == '__main__':

    print ('Start read data')
    time_1 = time.time()
    train_features, train_labels = readData('data/trainingDigits')
    test_features, test_labels = readData('data/testDigits')
    time_2 = time.time()
    print ('read data cost ',time_2 - time_1,' second','\n')

    print ('Start training')
    ada = AdaBoost()
    ada.train(train_features, train_labels)

    time_3 = time.time()
    print ('training cost ',time_3 - time_2,' second','\n')

    print ('Start predicting')
    test_predict = ada.predict(test_features)
    time_4 = time.time()
    print ('predicting cost ',time_4 - time_3,' second','\n')
    for i in range(len(test_labels)):
        print("The accuracy number:%d, the predict number: %d" % (test_labels[i],test_predict[i]))
    score = accuracy_score(test_labels,test_predict)
    print ("The accruacy socre is ", score)