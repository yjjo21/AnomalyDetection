

################################################################################

import os
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


class ArchiTools():
    def __init__(self, sHomeDir):
        self.homeDir = sHomeDir
        self.datDir = os.path.join(self.homeDir, 'dat')
        self.outDir = os.path.join(self.homeDir, 'out')


class DataTools():
    def __init__(self, sHomeDir):
        self.homeDir = sHomeDir
        self.datDir = os.path.join(self.homeDir, 'dat')
        self.outDir = os.path.join(self.homeDir, 'out')


    def fnLoadData(self):
        DF = pd.read_csv(self.datDir +  '/creditcard.csv')
        DF.drop('Time', axis = 1, inplace = True)
        DF['Class'] = DF['Class'].astype('category')

        self.yDF = DF[['Class']]
        self.xDF = DF.drop('Class', axis =1)
        print(self.xDF.shape, self.yDF.shape)


    def fnSplit(self):
        trainXNP, testXNP, trainYNP, testYNP = train_test_split(self.xDF,
                                                                self.yDF,
                                                                test_size = .2,
                                                                random_state = 0,
                                                                stratify = self.yDF)

        trainXNP, validXNP, trainYNP, validYNP = train_test_split(trainXNP,
                                                                  trainYNP,
                                                                  test_size = .3,
                                                                  random_state = 0,
                                                                  stratify = trainYNP)

        self.trainXNP, self.trainYNP = trainXNP, trainYNP
        self.validXNP, self.validYNP = validXNP, validYNP 
        self.testXNP , self.testYNP  = testXNP , testYNP


    def fnNormalize(self):
        standardScaler = StandardScaler()
        standardScaler.fit(self.trainXNP)
        self.trainNormXNP = standardScaler.transform(self.trainXNP)
        ### Need to Test Normalize
         




    def fnGenModel(self):
        self.lr_clf = LogisticRegression()
        self.lr_clf.fit(self.trainNormXNP, self.trainYNP) # , , max_iters = 200    


    def fnPrediction(self):
        self.yPred = self.lr_clf.predict(self.testXNP)







class ModelTools():
    def __init__(self, sModelName):
        self.modelName = sModelName

    def fnGenModel(self):
        lr_clf = LogisticRegression()
        lr_clf.fit(xTrain, yTrain) # , , max_iters = 200    


    def fnModelTrain(self, sModelName):
        model.fit(xTrain, yTrain)
        yPred = model.predict(xTest)


    def fnPrediction(self):
        yPredTest = lr_clf.predict(xTest)
        print(yPredTest.shape)



class EvaluationTools():
    def __init__(self, yRealNP, yPredTestNP, sModelName):
        self.yReal = yRealNP.reshape(-1)
        self.yPred = yPredTestNP
        self.modelName = sModelName
        ## save directory


    def fnMakeOutput(self):
        self.outputDF = pd.DataFrame({'model_name':self.modelName,
                                      'yReal':self.yReal, 
                                      'yPred':self.yPred})
        
        
    def fnMakeAccuracy(self):
        self.confusion = confusion_matrix(self.yReal, self.yPred).tolist()
        self.accuracy  = accuracy_score(self.yReal, self.yPred)
        self.precision = precision_score(self.yReal, self.yPred)
        self.recall    = recall_score(self.yReal, self.yPred)
        self.f1        = f1_score(self.yReal, self.yPred)

        self.accDF = pd.DataFrame({'confusion': [self.confusion],
                                   'accuracy' : round(self.accuracy, 4),
                                   'precision': round(self.precision,4),
                                   'recall'   : round(self.recall, 4),
                                   'f1'       : round(self.f1, 4)})


        print('')

        # need to save options


################################################################################

