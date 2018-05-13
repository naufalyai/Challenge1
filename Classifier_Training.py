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
def akurasi(actual,predictions):
    benar = 0
    salah = 0
    for i in range(np.shape(actual)[0]):
        if (actual[i] == predictions[i]):
            benar +=1
        else:
            salah+=1
    return benar/np.shape(actual)[0]

my_data = genfromtxt('cs-training.csv', delimiter=',')
featuredData = genfromtxt('cleanTraining.csv', delimiter=',')

process_data = my_data[1:,1:]

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(featuredData,process_data[:,0])

MLP = MLPClassifier(hidden_layer_sizes=(12,))
MLP = MLP.fit(featuredData,process_data[:,0])

NB = naive_bayes.GaussianNB()
NB.fit(featuredData,process_data[:,0])

prediction = NB.predict(featuredData)
prediction2 = clf.predict(featuredData)
prediction3 = MLP.predict(featuredData)

record = []
naive = f1_score(process_data[:,0],prediction,average='macro')
rf = f1_score(process_data[:,0],prediction2,average='macro')
mlp = f1_score(process_data[:,0],prediction3,average='macro')
print('Naive Bayes F1-Score : ',naive)
print('Random Forest F1-Score : ',rf)
print('MLP F1-Score : ', mlp)

print(akurasi(process_data[:,0],prediction))
print(akurasi(process_data[:,0],prediction2))
print(akurasi(process_data[:,0],prediction3))

tn1, fp1, fn1, tp1 = confusion_matrix(process_data[:,0],prediction).ravel()
tn2, fp2, fn2, tp2 = confusion_matrix(process_data[:,0],prediction2).ravel()
tn3, fp3, fn3, tp3 = confusion_matrix(process_data[:,0],prediction3).ravel()

record.append(['Naive Bayes', naive, tn1,fp1,fn1,tp1])
record.append(['Random Forest', rf, tn2,fp2,fn2,tp2])
record.append(['ANN (MLP)', mlp, tn3,fp3,fn3,tp3])

record = np.array(record)
df = pd.DataFrame(record)
df.to_excel('Record_without_mutualInformation.xlsx',header=['classifier','f1-score macro_average','true_negative','false_positive','false_negative','true_positive'])
