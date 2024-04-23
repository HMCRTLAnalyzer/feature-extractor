#!/bin/python3
# starter code gratefully borrowed from : https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
# https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html

# from numpy import loadtxt
import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import metrics
import time

# ----------------------------------------
# to do: try training on only non-memories! -diego's suggestion
# ----------------------------------------

def load_dataset_csv(file = 'test4_15.csv', predict=False, drop_mem=False):
    '''
    Loads output csv from feature extraction into dataframe for sklearn/xgboost learning into X, Y, and name dataframes
    '''
    # load data as pandas dataframe

    df = pd.read_csv(file)

    # Remove entries marked as memory
    if (drop_mem) and (not predict):
        df = df[df.memory == 0]

    # read and separate dataframe according to known dataset info column names

    dataset_info_names = ['module', 'delay_delta', 'area_delta', 'memory', 'sensitive'] # Hard coded, maybe turn this into a variable

    try: # Try block for full dataset
        name_list = list(df.columns.values)
        # separate dataframe based on whether a column's name is in the list of known dataset info column names
        dataset_names_list = [x for x in name_list if x in dataset_info_names]
        feature_names_list = [x for x in name_list if x not in dataset_info_names] 
        label_data = df[dataset_names_list]
        feature_data = df[feature_names_list]
        feature_names = list(feature_data.columns.values) # Cast feature names as numpy array
    except Exception as e:
        print(f"Error: Failed to read in Dataframe due to {e}, exiting!")
        return 1
    
    
    #save feature names out separately from rest of dataset
    # feature_names = dataset[0, start_of_data:] 
    # print(feature_names)
    # dataset = dataset[1:, :]
    if not predict:
        dataset_metadata = label_data
        dataset_metadata.drop('sensitive', axis=1)

    seed = 7
    # split data into train and test sets
    if predict:
        #keep all the data when we parse a sample
        X_train = feature_data
        y_train = label_data
        X_test = feature_data
        y_test = label_data #the test ones don't matter 
    else:
        test_size = 0.3
        X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=test_size, random_state=seed)

    if not (predict):
        # Filter out columns in labels not related to sensitivity. Store those in a separate dataframe
        name_test = y_test
        name_train = y_train
        name_test.drop('sensitive',axis=1)
        name_train.drop('sensitive',axis=1)
        y_test = y_test['sensitive']
        y_train = y_train['sensitive']
    else:
        name_test = y_test
        name_train = y_train
    
    #save everything for quicker access
    return X_train, X_test, y_train, y_test, name_train, name_test, feature_names

def load_dataset(df, predict=False, drop_mem=False):
    """
        Loads a dataset from a pandas dataframe. Essentially a copy of load_dataset_csv.

        Automatically splits the dataframe
    """
    # Remove entries marked as memory
    if (drop_mem) and (not predict):
        df = df[df.memory == 0]

    # read and separate dataframe according to known dataset info column names
    dataset_info_names = ['module', 'delay_delta', 'area_delta', 'memory', 'sensitive'] # Hard coded, maybe turn this into a variable

    try: # Try block for full dataset
        name_list = list(df.columns.values)
        # separate dataframe based on whether a column's name is in the list of known dataset info column names
        dataset_names_list = [x for x in name_list if x in dataset_info_names]
        feature_names_list = [x for x in name_list if x not in dataset_info_names] 
        label_data = df[dataset_names_list]
        feature_data = df[feature_names_list]
        feature_names = list(feature_data.columns.values) # Cast feature names as numpy array
    except Exception as e:
        print(f"Error: Failed to read in Dataframe due to {e}, exiting!")
        return 1
    
    
    #save feature names out separately from rest of dataset
    # feature_names = dataset[0, start_of_data:] 
    # print(feature_names)
    # dataset = dataset[1:, :]
    if not predict:
        dataset_metadata = label_data
        dataset_metadata.drop('sensitive', axis=1)

    seed = 7
    # split data into train and test sets
    if predict:
        #keep all the data when we parse a sample
        X_train = feature_data
        y_train = label_data
        X_test = feature_data
        y_test = label_data #the test ones don't matter 
    else:
        test_size = 0.3
        X_train, X_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=test_size, random_state=seed)

    if not (predict):
        # Filter out columns in labels not related to sensitivity. Store those in a separate dataframe
        name_test = y_test
        name_train = y_train
        name_test.drop('sensitive',axis=1)
        name_train.drop('sensitive',axis=1)
        y_test = y_test['sensitive']
        y_train = y_train['sensitive']
    else:
        name_test = y_test
        name_train = y_train
    
    #save everything for quicker access
    return X_train, X_test, y_train, y_test, name_train, name_test, feature_names

def predict_model(input_csv, model):
    """
        Takes in a model and an input csv

        outputs a set of predictions according to xgboost based on the input csv
    """
    X_train_pd, X_test_pd, y_train_pd, y_test_pd, name_train, name_test, feature_names = \
        load_dataset_csv(input_csv, predict=True)
    X = X_train_pd.to_numpy()

    prediction = model.predict(X)

    return prediction, name_train.values.tolist()


def plot_feature_importance_to_file(fileloc, model, N, feature_names):
    """
        Takes in an XGBoost model, number of features to display N, and a list of feature names
        Write a graph of feature importance for top N features to fileloc

        Returns nothing
    """
    # feat importance with names f1,f2,...
    axsub = xgb.plot_importance(model, max_num_features=N)

    # get the original names back
    Text_yticklabels = list(axsub.get_yticklabels())
    myfeatures = list(feature_names)
    dict_features = dict(enumerate(myfeatures))
    lst_yticklabels = [ Text_yticklabels[i].get_text().lstrip('f') for i in range(len(Text_yticklabels))]
    lst_yticklabels = [ dict_features[int(i)] for i in lst_yticklabels]

    axsub.set_yticklabels(lst_yticklabels)
    plt.tight_layout()
    plt.savefig(fileloc,  dpi = 600)

    return


# -----------------------------------------------------------------------------------------------------------------
# main
X_train_pd, X_test_pd, y_train_pd, y_test_pd, name_train, name_test, feature_names = \
    load_dataset_csv('/home/nlucio/feature-extractor/processed_data/temp_top500_LEnorm.csv',drop_mem=True)

plot_folder = "pretty_pictures/"
model_folder = "models/"
dt = time.strftime("%Y-%d_%H%M")

importance_plot_loc = plot_folder+"xgboost_importance"+dt+".png"
importance_N = 15

# Cast X and Y vectors to numpy arrays.

# make model and do cross validation
# this model uses the sklearn estimator interface *note that this is different from the native interface!!*
clf = xgb.XGBClassifier(tree_method="hist")

# making a different model to try early stopping
# print("early stop model ----------------------------------------")
# clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=100)

# Cast X and Y to numpy arrays.

X_test = X_test_pd.to_numpy()
X_train = X_train_pd.to_numpy()
y_test = y_test_pd.to_numpy()
y_train = y_train_pd.to_numpy()

# reuniting names and features
clf.feature_types = None
clf.feature_names = list(feature_names)

# Train model

clf.fit(X_train, y_train, eval_set=[(X_test, y_test)])
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate cross validation and model
scores = cross_val_score(clf, X_train, y_train, cv=5)
print(scores) #[0.81730769 0.84615385 0.86538462 0.85576923 0.83653846] before any training
print(clf.get_params())
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print("train score: " + str(train_score))
print("test score: " + str(test_score)) 
#train score: 0.9730769230769231 test score: 0.874439461883408

# Save model into models folder

clf.save_model(model_folder+"xgb_model_"+dt+".model")

#if we have a single unknown sample and a trained model, can get the prediction via the following:
ynew, names = predict_model("/home/nlucio/feature-extractor/processed_data/openMSP430_real.csv", clf)
for i in range(len(ynew)):
    print("module: %s, Predicted=%s" % (names[i], ynew[i]))

# plots of tree
a = xgb.plot_tree(clf, num_trees = 1)
plt.savefig(plot_folder+"xgboost_early_stop_tree_"+dt+".png", dpi = 600)

plot_feature_importance_to_file(importance_plot_loc, clf, importance_N, feature_names)

# # feat importance with names f1,f2,...
# axsub = xgb.plot_importance(clf, max_num_features=15)

# # get the original names back
# Text_yticklabels = list(axsub.get_yticklabels())
# myfeatures = list(feature_names)
# dict_features = dict(enumerate(myfeatures))
# lst_yticklabels = [ Text_yticklabels[i].get_text().lstrip('f') for i in range(len(Text_yticklabels))]
# lst_yticklabels = [ dict_features[int(i)] for i in lst_yticklabels]

# axsub.set_yticklabels(lst_yticklabels)
# plt.tight_layout()
# plt.savefig(plot_folder+"xgboost_importance_early_stop_"+dt+".png",  dpi = 600)

# Save model into models folder

clf.save_model(model_folder+"xgb_model_"+dt+".model")


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
