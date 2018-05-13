from numpy import genfromtxt
import numpy as np
from sklearn.metrics import confusion_matrix
from pandas import read_csv
from sklearn import naive_bayes
from sklearn.metrics import f1_score
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pandas as pd

my_data = genfromtxt('cs-training.csv', delimiter=',')
featuredData = genfromtxt('cleanTraining.csv', delimiter=',')
process_data = my_data[1:,1:]
featuredTest = genfromtxt('cleanTesting.csv', delimiter=',')

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(featuredData,process_data[:,0])
prediction2 = np.array(clf.predict_proba(featuredData),dtype=float)

df = pd.DataFrame(prediction2[:,1])
df.to_excel('RecordTest_RandomForest.xlsx',header=['probability'])