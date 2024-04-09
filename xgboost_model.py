# starter code gratefully borrowed from : https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/

from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier
import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data as numpy array
dataset = loadtxt('/home/qualcomm_clinic/RTL_dataset/temp_top350.csv', delimiter=",", skiprows = 1, dtype="str")
dataset = dataset[:, 1:-1].astype(np.float64) #drop module names and last column b/c brackets
# split data into X and y
X = dataset[:,3:]
Y = dataset[:,0] # sensitivity in 0, memory in 1
# split data into train and test sets
seed = 7
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#save images
a = xgboost.plot_tree(model, num_trees = 1)
plt.savefig("xbg_tree.png", dpi = 600)
b = xgboost.plot_importance(model, max_num_features=20)
plt.savefig("xbg_importance.png", dpi = 600)