import pylab as pl
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#getiing data from files
x_train_file = 'x_train.xlsx'
y_train_file = 'y_train.xlsx'
x_test_file = 'x_test.xlsx'
y_test_file = 'y_test.xlsx'
import ConfMat


def my_data(file_name):

    data = np.array(pd.read_excel(file_name, 'Sheet1'))
    scaler=preprocessing.MinMaxScaler()
    data=scaler.fit_transform(data)
    data=data.astype('float32')

    return data
'''
x_train=my_data(x_train_file)
x_test=my_data(x_test_file)
y_train=my_data(y_train_file)
y_test=my_data(y_test_file)
'''

#spilit data to test and train
x_train, x_test, y_train, y_test= train_test_split( my_data(x_train_file),
                                                     my_data(y_train_file),
                                                     test_size=0.05)


######## SVM ########
rf = svm.SVC(verbose=True)
# Train the model on training data
rf.fit(x_train, y_train)

# predict method on the test data
predictions = rf.predict(x_test)

cm=confusion_matrix(y_test,predictions)
acc=((cm[0,0]+cm[1,1])/np.size(predictions))*100
print("\nTest Accuracy = " + format(acc, '.2f') + "%")

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
ConfMat.plot_confusion_matrix(cnf_matrix, classes=['Surviability', 'Surviability'],
                              title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
ConfMat.plot_confusion_matrix(cnf_matrix, classes=['Surviability', 'Surviability'],
                              normalize=True, title='Normalized confusion matrix')

plt.show()

######## Random forest ########


rfc = RandomForestClassifier(n_estimators=100)
# Train the model on training data
rfc.fit(x_train, y_train)

# Use the forest's predict method on the test data
predictions = rfc.predict(x_test)

cm=confusion_matrix(y_test,predictions)
acc=((cm[0,0]+cm[1,1])/np.size(predictions))*100
print("\nTest Accuracy = " + format(acc, '.2f') + "%")

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
ConfMat.plot_confusion_matrix(cnf_matrix, classes=['Surviability', 'Surviability'],
                              title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
ConfMat.plot_confusion_matrix(cnf_matrix, classes=['Surviability', 'Surviability'],
                              normalize=True, title='Normalized confusion matrix')

plt.show()

######## DecisionTreeClassifier ########

dt = DecisionTreeClassifier (criterion = 'gini', random_state = 100,max_depth=3, min_samples_leaf=5)
# Train the model on training data
dt.fit(x_train, y_train)

# predict method on the test data
predictions = dt.predict(x_test)

cm=confusion_matrix(y_test,predictions)
acc=((cm[0,0]+cm[1,1])/np.size(predictions))*100
print("\nTest Accuracy = " + format(acc, '.2f') + "%")

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predictions)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
ConfMat.plot_confusion_matrix(cnf_matrix, classes=['Surviability', 'Surviability'],
                              title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
ConfMat.plot_confusion_matrix(cnf_matrix, classes=['Surviability', 'Surviability'],
                              normalize=True, title='Normalized confusion matrix')

plt.show()
