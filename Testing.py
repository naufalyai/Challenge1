from sklearn.preprocessing import Imputer
import numpy as np
from numpy import genfromtxt
from keras import  models
my_test = genfromtxt('cs-test.csv', delimiter=',')
process_test = my_test[1:,2:]
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp = imp.fit(process_test)
clean_test = np.array(imp.transform(process_test))
print(clean_test.shape)
model = models.load_model('keras_model.h5')
pred = model.predict_proba(clean_test)
pred2 = model.predict(clean_test)
# print(pred)
np.savetxt('probability.txt',pred)
np.savetxt('class.txt',pred2)
# for i in range(np.shape(pred)[0]):
#     print(i,pred[i])