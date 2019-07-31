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
from sklearn import model_selection
from sklearn.calibration import calibration_curve
from IPython import get_ipython


#getiing data from files
x_train_file = 'x_train.xlsx'
y_train_file = 'y_train.xlsx'
x_test_file = 'x_test.xlsx'
y_test_file = 'y_test.xlsx'

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
                                                     shuffle=True,
                                                     test_size=0.05)



svm_classification = svm.SVC(probability=True)
# Train the model on training data
svm_model= svm_classification.fit(x_train, y_train)
# Use the forest's predict method on the test data
svm_predictions = svm_model.predict_proba(x_test)




rfc = RandomForestClassifier(n_estimators=100)
# Train the model on training data
rfc_model= rfc.fit(x_train, y_train)
# Use the forest's predict method on the test data
rfc_predictions = rfc_model.predict_proba(x_test)

dt = DecisionTreeClassifier ( random_state = 100, min_samples_leaf=50)
# Train the model on training data
dt.fit(x_train, y_train)

# Use the forest's predict method on the test data
dt_predictions = dt.predict_proba(x_test)


####### MLP #########
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization, Dropout, Embedding, Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras import regularizers


#making our model
model= Sequential([

    Dense(100, input_shape=(23,),kernel_initializer='random_normal',
          kernel_regularizer=regularizers.l1(0.01)),
    Dropout(0.3),
    Activation('relu'),
    BatchNormalization(),
    Dense(32, kernel_initializer='random_normal',
          kernel_regularizer=regularizers.l1(0.01)),
    Dropout(0.3),
    Activation('relu'),
    BatchNormalization(),
    Dense(2,activation='softmax')
    ])

#Training with training data
optimizer = RMSprop(lr=3e-5, decay=1e-6)
model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
#model.fit_gene0rator((my_data(x_train_file),my_data(y_train_file)), )
history=model.fit(x_train,y_train, validation_split=0.06, batch_size=128,
          epochs=300, shuffle=True, verbose=0)

mlp_predictions= model.predict(x_test)

svm_y, svm_x = calibration_curve(y_test, svm_predictions[:,1], n_bins=8)
rfc_y, rfc_x = calibration_curve(y_test, rfc_predictions[:,1], n_bins=8)
dt_y, dt_x = calibration_curve(y_test, dt_predictions[:,1], n_bins=8)
mlp_y, mlp_x = calibration_curve(y_test, mlp_predictions[:,1], n_bins=8)

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

fig, ax = plt.subplots()

plt.plot(svm_x, svm_y, marker='o', linewidth=1, label='SVM')
plt.plot(rfc_x, rfc_y, marker='o', linewidth=1, label='RF')
plt.plot(dt_x, dt_y, marker='o', linewidth=1, label='DT')
plt.plot(mlp_x, mlp_y, marker='o', linewidth=1, label='MLP')

# reference line, legends, and axis labels
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
fig.suptitle('Calibration plot for Breast Cancer data')
ax.set_xlabel('Predicted probability')
ax.set_ylabel('True probability in each bin')
plt.legend()
plt.show()



