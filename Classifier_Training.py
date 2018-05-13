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

# Manually accuracy function
def akurasi(actual,predictions):
    benar = 0
    salah = 0
    for i in range(np.shape(actual)[0]):
        if (actual[i] == predictions[i]):
            benar +=1
        else:
            salah+=1
    return benar/np.shape(actual)[0]

# Load data train
my_data = genfromtxt('cs-training.csv', delimiter=',')
featuredData = genfromtxt('cleanTraining.csv', delimiter=',')
process_data = my_data[1:,1:]

# define Random Forest Classifier and train it. I used 10 tree estimator for this case
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(featuredData,process_data[:,0])

# define ANN MLP Classifier and train it. I used 12 neuron in hidden layers, adam optimizer, and 200 epoch.
MLP = MLPClassifier(hidden_layer_sizes=(12,))
MLP = MLP.fit(featuredData,process_data[:,0])

# define Naive Bayes Classifier. I used Gaussian Naive Bayes because the data is mostly continous
NB = naive_bayes.GaussianNB()
NB.fit(featuredData,process_data[:,0])

# Predict training data using 3 classifier
prediction = NB.predict(featuredData)
prediction2 = clf.predict(featuredData)
prediction3 = MLP.predict(featuredData)

# Performance measure, I used macro average f1-measure
record = []
naive = f1_score(process_data[:,0],prediction,average='macro')
rf = f1_score(process_data[:,0],prediction2,average='macro')
mlp = f1_score(process_data[:,0],prediction3,average='macro')
print('Naive Bayes F1-Score : ',naive)
print('Random Forest F1-Score : ',rf)
print('MLP F1-Score : ', mlp)

# I try to compare with manually check accuracy
print(akurasi(process_data[:,0],prediction))
print(akurasi(process_data[:,0],prediction2))
print(akurasi(process_data[:,0],prediction3))

# Confusion matrix to check true positive, false positive, true negative, and false negative
tn1, fp1, fn1, tp1 = confusion_matrix(process_data[:,0],prediction).ravel()
tn2, fp2, fn2, tp2 = confusion_matrix(process_data[:,0],prediction2).ravel()
tn3, fp3, fn3, tp3 = confusion_matrix(process_data[:,0],prediction3).ravel()

# save all measure into array
record.append(['Naive Bayes', naive, tn1,fp1,fn1,tp1])
record.append(['Random Forest', rf, tn2,fp2,fn2,tp2])
record.append(['ANN (MLP)', mlp, tn3,fp3,fn3,tp3])

# save the record into a excel file
record = np.array(record)
df = pd.DataFrame(record)
df.to_excel('Record_without_mutualInformation.xlsx',header=['classifier','f1-score macro_average','true_negative','false_positive','false_negative','true_positive'])
