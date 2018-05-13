from numpy import genfromtxt
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
my_data = np.array(genfromtxt('cleanTraining.csv', delimiter=','))
my_test = np.array(genfromtxt('cleanTesting.csv', delimiter=','))
MI = np.array(genfromtxt('Mutual_Information.csv', delimiter=','))
rata = np.mean(MI[:,1])
MI2 = np.array(MI[MI[:,1] >= rata])
print(MI2)
attribute_name = ['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']
used_attribute = []
for i in range(np.shape(MI2)[0]):
    used_attribute.append(attribute_name[int(MI2[i,0])])
print(used_attribute)
