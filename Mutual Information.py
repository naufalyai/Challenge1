from numpy import genfromtxt
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import mutual_info_classif


my_data = genfromtxt('cs-training.csv', delimiter=',')
my_test = genfromtxt('cs-test.csv', delimiter=',')
process_data = my_data[1:,1:]
process_test = my_test[1:,1:]
attribute_name = ['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']
print(len(attribute_name))
process_test = my_test[1:,1:]
imp1 = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp1 = imp1.fit(process_data[:,1:])
imp2 = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp2 = imp2.fit(process_test[:,1:])
clean_data = np.array(imp1.transform(process_data[:,1:]))
clean_test = np.array(imp2.transform(process_test[:,1:]))
MI = mutual_info_classif(clean_data,process_data[:,0])
MI2 = []
for i in range(np.shape(MI)[0]):
    MI2.append([i,MI[i]])
MI2 = np.array(MI2)
saver2 = pd.DataFrame(clean_data)
saver2.to_csv('cleanTraining.csv',header=False,index=False)
saver3= pd.DataFrame(clean_test)
saver3.to_csv('cleanTesting.csv',header=False,index=False)
saver = pd.DataFrame(MI2)
saver.to_csv('Mutual_Information.csv',header=False,index=False)