import matplotlib
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import keras
from keras import regularizers
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from keras import backend as k
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization, Dropout, Embedding, Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import sparse_categorical_crossentropy
from keras import regularizers
from keras.models import load_model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
import ConfMat


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


'''
y_train=y_train.astype(int)
y_train = np_utils.to_categorical(y_train, 2)


y_test=y_test.astype(int)
y_test = np_utils.to_categorical(y_test, 2)
'''
#making our model
model= Sequential([

    Dense(100, input_shape=(12,),kernel_initializer='random_normal',
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
model.summary()
#Training with training data
optimizer = RMSprop(lr=3e-5, decay=1e-6)
model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
#model.fit_gene0rator((my_data(x_train_file),my_data(y_train_file)), )
history=model.fit(x_train,y_train, validation_split=0.06, batch_size=128,
          epochs=300, shuffle=True, verbose=2)

pred=model.predict_classes(x_test,batch_size=20,verbose=0)

score = model.evaluate(x_test, y_test, verbose=2)
print("Test Accuracy = " + format(score[1]*100, '.2f') + "%")

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'],'r')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'],'r')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#save model
model.save('BestSurvivalshipModel.h5')
'''
#load model
from keras.models import load_model
new_model=load_model('BestSurvivalshipModel.h5')
'''

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
ConfMat.plot_confusion_matrix(cnf_matrix, classes=['Surviability <5 years', 'Surviability >5 years'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
ConfMat.plot_confusion_matrix(cnf_matrix, classes=['Surviability <5 years', 'Surviability >5 years'],
                      normalize=True, title='Normalized confusion matrix')


plt.show()
