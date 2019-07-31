import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier

x_train_file = 'x_train.xlsx'
y_train_file = 'y_train.xlsx'



feat = pd.read_excel('x_train.xlsx')
features=list(feat)
print(features)

def my_data(file_name):

    data = np.array(pd.read_excel(file_name, 'Sheet1'))
    scaler=preprocessing.MinMaxScaler()
    data=scaler.fit_transform(data)
    data=data.astype('float32')

    return data
X=my_data(x_train_file)
Y=my_data(y_train_file)


# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, Y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f]+1, importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices+1)
plt.xlim([-1, X.shape[1]])
plt.show()
