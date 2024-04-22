# starter code gratefully borrowed from : https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
# https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html

from numpy import loadtxt
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import metrics

# ----------------------------------------
# to do: try training on only non-memories! -diego's suggestion
# ----------------------------------------

def load_feature_csv_into_npz(file = 'test4_15.csv', clean_file_name = 'clean_data', predict = False):
    '''
    Loads output csv from feature extraction into usuable numpy arrays for sklearn/xgboost learning and saves it into a .npz file for quick access
    Can also process data that we want to make a prediction about
    '''
    # load data as numpy array

    if (predict): #all predictions will be made for the 350 columns
        print("cutting data for predictions")
        start_of_data = 2
        name_col = 0
        sens_col = 0
        mem_col = 0 #these are just numbers so we're fine, not going to be used anyway
    elif (file == "test4_15.csv"): #500 columns each, with delta cols
        print("using new data")
        start_of_data = 5
        name_col = 0
        sens_col = 3
        mem_col = 4
    else: #350 cols
        start_of_data = 3
        name_col = 0
        sens_col = 1
        mem_col = 2
    #dataset = loadtxt('/home/qualcomm_clinic/RTL_dataset/temp_top350.csv', delimiter=",", skiprows = 1, dtype="str")
    dataset = loadtxt(file, delimiter=",", skiprows = 0, dtype="str")
    
    # split data into X and y
    if predict: #drop random extra indexing col
        dataset = dataset[:, 1:]
    
    #save feature names out separately from rest of dataset
    feature_names = dataset[0, start_of_data:] 
    print(feature_names)
    dataset = dataset[1:, :]


    X = dataset[:, start_of_data:]
    X[:, -1] = [name.strip("[]") for name in X[:, -1]]
    Y = dataset[:, name_col:start_of_data] # columns in order: names, sensitivity, memory OR names, delay delta , area delta, sens, mem
    Y[:, name_col] = [name.strip("[]") for name in Y[:, name_col]]

    seed = 7
    # split data into train and test sets
    if predict:
        #keep all the data when we parse a sample
        X_train = X
        y_train = Y
        X_test = X
        y_test = Y #the test ones don't matter 
    else:
        test_size = 0.3
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    #print(X_train)
    #print(y_train)

    #to keep labels with data, pull off only after splitting
    names_train = y_train[:, name_col]
    # sens_train = y_train[:, sens_col]
    # mem_train = y_train[:, mem_col]
    names_test = y_test[:, name_col]
    # sens_test = y_test[:, sens_col]
    # mem_test = y_test[:, mem_col]

    #clean up data now (y's should be 1D)
    X_train = X_train[:, start_of_data:].astype(np.float64)
    X_test = X_test[:, start_of_data:].astype(np.float64)
    if not(predict):
        y_train = y_train[:, sens_col].astype(np.float64) #label = sensitivity so column 1
        y_test = y_test[:, sens_col].astype(np.float64)

    #save everything for quicker access
    np.savez(clean_file_name, X_train = X_train, X_test=X_test, y_train=y_train, y_test=y_test, 
             names_train= names_train, names_test = names_test,
             feature_names = feature_names)



# -----------------------------------------------------------------------------------------------------------------
# main
#load_feature_csv_into_npz('/home/qualcomm_clinic/RTL_dataset/temp_top350.csv', clean_file_name="clean_data")
npz = np.load("clean_data.npz")
X_train = npz["X_train"]
X_test = npz["X_test"]
y_train = npz["y_train"]
y_test = npz["y_test"]
names_train = npz["names_train"]
names_test = npz["names_test"]
feature_names = npz["feature_names"] #these feature names are from the training data

# make model and do cross validation
# this model uses the sklearn estimator interface *note that this is different from the native interface!!*
""" clf = xgb.XGBClassifier(tree_method="hist")
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores) #[0.81730769 0.84615385 0.86538462 0.85576923 0.83653846] before any training
print(clf.get_params()) """

# making a different model to try early stopping
print("early stop model ----------------------------------------")
earlystopmodel = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=10)

# reuniting names and features
earlystopmodel.feature_types = None
earlystopmodel.feature_names = list(feature_names)
#print(earlystopmodel.feature_names)

#train model and evaluate
earlystopmodel.fit(X_train, y_train, eval_set=[(X_test, y_test)])
y_pred = earlystopmodel.predict(X_test)
predictions = [round(value) for value in y_pred]

train_score = earlystopmodel.score(X_train, y_train)
test_score = earlystopmodel.score(X_test, y_test)
print("train score: " + str(train_score))
print("test score: " + str(test_score)) 
#train score: 0.9730769230769231 test score: 0.874439461883408

# yet another sklearn estimator but this one does a sweep
""" clf = GridSearchCV(
        xgb.XGBClassifier(),
        {"max_depth": [4, 6, 8, 10], 
         "n_estimators": [50, 100, 200],
         "learning_rate": [0.1, 0.05, 0.2],
        },
        verbose=1,
        n_jobs=4,
    )
clf.fit(X_train, y_train)
print(clf.best_score_)
print(clf.best_params_)  """

#if we have a single unknown sample and a trained model, can get the prediction via the following:
load_feature_csv_into_npz("/home/nlucio/feature-extractor/processed_data/openMSP430_features.csv", clean_file_name="openMSP430.npz", predict = True)
npz = np.load("openMSP430.npz")
Xnew = npz["X_train"]
namesnew = npz["names_train"]
ynew = earlystopmodel.predict(Xnew)
for i in range(len(Xnew)):
    print("module: %s, X=%s, Predicted=%s" % (namesnew[i],Xnew[i], ynew[i]))

#plots of tree
#a = xgb.plot_tree(earlystopmodel, num_trees = 1)
#plt.savefig("xgboost_early_stop_tree.png", dpi = 600)

# Visualize feature importance
#b = xgb.plot_importance(earlystopmodel, max_num_features=20)
# create dict to use later
myfeatures = list(feature_names)
dict_features = dict(enumerate(myfeatures))

# feat importance with names f1,f2,...
axsub = xgb.plot_importance(earlystopmodel, max_num_features=10)

# get the original names back
Text_yticklabels = list(axsub.get_yticklabels())
dict_features = dict(enumerate(myfeatures))
lst_yticklabels = [ Text_yticklabels[i].get_text().lstrip('f') for i in range(len(Text_yticklabels))]
lst_yticklabels = [ dict_features[int(i)] for i in lst_yticklabels]

axsub.set_yticklabels(lst_yticklabels)
print(dict_features)
plt.savefig("xgboost_importance_early_stop.png",  dpi = 600)



# https://machinelearningmastery.com/make-predictions-scikit-learn/

# QUESTION FOR GOKCE: which parameters should we even be training for????? what even is n_estimators???
# prob just booster = gbtree is fine
'''
terminal printout 
0.8461538461538461
{'booster': 'gbtree', 'learning_rate': 0.2, 'max_depth': 4, 'n_estimators': 100}
(penguin) [esundheim@tera feature-extractor]$ python xgb_crossval.py 
using new data
Fitting 5 folds for each of 36 candidates, totalling 180 fits
0.8461538461538461
{'learning_rate': 0.2, 'max_depth': 4, 'n_estimators': 100}'''
