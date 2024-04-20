# starter code gratefully borrowed from : https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/

from numpy import loadtxt
import numpy as np
from xgboost import XGBClassifier
import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics

# load data as numpy array
file = 'test4_15.csv'
if (file == "test4_15.csv"):
    print("using new data")
    start_of_data = 5
    sens_col = 3
    mem_col = 4
else:
    start_of_data = 3
    sens_col = 1
    mem_col = 2
#dataset = loadtxt('/home/qualcomm_clinic/RTL_dataset/temp_top350.csv', delimiter=",", skiprows = 1, dtype="str")
dataset = loadtxt('test4_15.csv', delimiter=",", skiprows = 1, dtype="str")

# split data into X and y
X = dataset[:, start_of_data:]
X[:, -1] = [name.strip("[]") for name in X[:, -1]]
Y = dataset[:, 0:start_of_data] # columns in order: names, sensitivity, memory OR names, delay delta , area delta, sens, mem
Y[:, 0] = [name.strip("[]") for name in Y[:, 0]]

# ----------------------------------------
# to do: try training on only non-memories! -diego's suggestion
# ----------------------------------------
# split data into train and test sets
seed = 7
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#to keep labels with data, pull off only after splitting
names_train = y_train[:, 0]
sens_train = y_train[:, sens_col]
mem_train = y_train[:, mem_col]
names_test = y_test[:, 0]
sens_test = y_test[:, sens_col]
mem_test = y_test[:, mem_col]

#clean up data now
X_train = X_train[:, start_of_data:].astype(np.float64)
X_test = X_test[:, start_of_data:].astype(np.float64)
y_train = y_train[:, sens_col].astype(np.float64) #label = sensitivity so column 1
y_test = y_test[:, sens_col].astype(np.float64)

#cleaning up data to try fewer columns
num_features = 350
X_train = np.concatenate((X_train[:, start_of_data:start_of_data+num_features], X_train[:, start_of_data+500: start_of_data+500+num_features], X_train[:, start_of_data+1000: start_of_data+1000+num_features], X_train[:, start_of_data+1500: start_of_data+1500+num_features], X_train[:,start_of_data+2000:start_of_data+2000+num_features]), axis = 1)
print(X_train)
X_test = np.concatenate((X_test[:, start_of_data:start_of_data+num_features], X_test[:, start_of_data+500: start_of_data+500+num_features], X_test[:, start_of_data+1000: start_of_data+1000+num_features], X_test[:, start_of_data+1500: start_of_data+1500+num_features], X_test[:,start_of_data+2000:start_of_data+2000+num_features]), axis = 1)

# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# error analysis - which ones are still being misclassified?
# false_pos = [names_test[i] for i in range(len(names_test)) if predictions[i] > y_test[i]]
# false_neg = [names_test[i] for i in range(len(names_test)) if predictions[i] < y_test[i]]
# print("False positives: \n")
# print(*false_pos, sep ='\n')
# print("False negatives: \n")
# print(*false_neg, sep ='\n')

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

a = xgboost.plot_tree(model, num_trees = 1)
#plt.savefig("xbg_tree.png", dpi = 600)
b = xgboost.plot_importance(model, max_num_features=20)
#plt.savefig("xbg_importance_" + str(num_features) + ".png", dpi = 600)