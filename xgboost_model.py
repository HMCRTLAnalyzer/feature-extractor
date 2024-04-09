# starter code gratefully borrowed from : https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/

from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier
import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

# load data as numpy array
dataset = loadtxt('/home/qualcomm_clinic/RTL_dataset/temp_top350.csv', delimiter=",", skiprows = 1, dtype="str")

# split data into X and y
X = dataset[:,3:]
X[:, -1] = [name.strip("[]") for name in X[:, -1]]
Y = dataset[:,0:2] # columns in order: names, sensitivity, memory
Y[:, 0] = [name.strip("[]") for name in Y[:, 0]]

# split data into train and test sets
seed = 7
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#to keep labels with data, pull off only after splitting
names_train = y_train[:, 0]
sens_train = y_train[:, 1]
mem_train = y_train[:, 2]
names_test = y_test[:, 0]
sens_test = y_test[:, 1]
mem_test = y_test[:, 2]

#clean up data now
X_train = X_train[:, 3:].astype(np.float64) #start at 3rd index and drop the last one
X_test = X_test[:, 3:].astype(np.float64)
y_train = y_train[:, 1].astype(np.float64) #label = sensitivity so column 1
y_test = y_test[:, 1].astype(np.float64)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

misclassified = [names_test[i] for i in range(len(names_test)) if predictions[i] != y_test[i]]
print(misclassified)

# evaluate predictions
accuracy = metrics.accuracy_score(y_test, predictions)
precision = metrics.precision_score(y_test, predictions)
sensitivity = metrics.recall_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Precision: %.2f%%" % (precision * 100.0))
print("Sensitivity: %.2f%%" % (sensitivity * 100.0))

#confusion matrix
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.savefig("xgboost_confusion_matrix.png")

#save images
'''
a = xgboost.plot_tree(model, num_trees = 1)
plt.savefig("xbg_tree.png", dpi = 600)
b = xgboost.plot_importance(model, max_num_features=20)
plt.savefig("xbg_importance.png", dpi = 600)'''