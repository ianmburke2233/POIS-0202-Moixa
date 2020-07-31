# Axiom Consulting Partners - SIOP Machine Learning Competition 2020

##################################
# Package Import
##################################
import sys
import csv
import math
import numpy as np
import pandas as pd
from operator import itemgetter
import time
import datetime
import copy
import re
from urllib.parse import quote_plus
import xlsxwriter

import matplotlib.pyplot as plt
import pyodbc as odbc
import sqlalchemy as sqla

#from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
#from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, chi2
from sklearn import metrics
from sklearn.preprocessing import KBinsDiscretizer, scale

# Handle annoying warnings
import warnings
import sklearn.exceptions


# Future
warnings.simplefilter(action='ignore', category=FutureWarning)

# Convergence 
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


##################################
# Database Connection
##################################

# connStr = (r'Driver={ODBC Driver 13 for SQL Server};'
#            r'Server=tcp:axiom-siop-2020.database.windows.net,1433;'
#            r'Database=SIOP 2020;'
#            r'Uid=ian.burke;'
#            r'Pwd=@zIFdX4#jey1;'
#            r'Encrypt=yes;'
#            r'TrustServerCertificate=no;')
# quoted = quote_plus(connStr)
# new_con = 'mssql+pyodbc:///?odbc_connect={}'.format(quoted)
# engine = sqla.create_engine(new_con, fast_executemany = True)
# conn = odbc.connect("Driver={ODBC Driver 13 for SQL Server};"
#                     "Server=tcp:axiom-siop-2020.database.windows.net,1433;"
#                     "Database=SIOP 2020;"
#                     "Uid=ian.burke;"
#                     "Pwd=@zIFdX4#jey1;"
#                     "Encrypt=yes;")
# cursor = conn.cursor()

##################################
# Globals
##################################
modelid = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp())  # Unique ID for a given model run

train_filename = 'scoredTrainingData'  # Name of the file used to train the model
test_filename = 'DevelopmentData'  # Name of the file used to test the model
first_feat_index = 9  # Column index of the first feature
feat_start = 0
last_feat_index = -1  # Column index of the last feature
target_var = 'Retained'  # Name of the target column
prot_var = 'Protected_Group'  # Name of the protected group column
label = 'UNIQUE_ID'  # Name of the column providing row labels (e.g. Name, ID number, Client Number)

model_type = 'LR'  # Type of model to be run (LR, DT, RF, GB, ADA, MLP, SVC)
grid_search = 0  # Control Switch for hyperparameter grid search
hp_grid = {'bootstrap': [True],
           'max_depth': [20],
           'max_features': ['sqrt'],
           'min_samples_split': [4],
           'min_samples_leaf': [4],
           'n_estimators': [1000]}  # Dictionary with hyperparameters to use in grid search

undersample = 0  # Control Switch for Down Sampling
us_type = 3  # Under Sampling type (1:RuS, 2:Near-Miss, 3:Cluster Centroids)
oversampling = 0  # Control Switch for Up Sampling
os_type = 1  # Over Sampling type (1:RoS, 2:SMOTE, 3:ADASYN)
cross_val = 0  # Control Switch for CV
norm_target = 0  # Normalize target switch
norm_features = 0  # Normalize features switch
binning = 0  # Control Switch for Bin Target
bin_cnt = 2  # If bin target, this sets number of classes
feat_select = 0  # Control Switch for Feature Selection
fs_type = 4  # Feature Selection type (1=Stepwise Backwards Removal, 2=Wrapper Select, 3=Univariate Selection)
lv_filter = 0  # Control switch for low variance filter on features
k_cnt = 5  # Number of 'Top k' best ranked features to select, only applies for fs_types 1 and 3


##################################
# Helper Functions
##################################


# Wrapper Select
def feat_space_search(arr, curr_idx):
    """Setup for exhaustive search currently. Can reformat to run Greedy, Random or Genetic"""
    global roll_idx, combo_ctr, best_score, sel_idx

    if curr_idx == feat_cnt:
        # If end of feature array, roll thru combinations
        roll_idx = roll_idx + 1
        print("Combos Searched so far:", combo_ctr, "Current Best Score:", best_score)
        for i in range(roll_idx, len(arr)):
            arr[i] = 0
        if roll_idx < feat_cnt - 1:
            feat_space_search(arr, roll_idx + 1)  # Recurse till end of rolls

    else:
        # Else setup next feature combination and calc performance
        arr[curr_idx] = 1
        data = data_np  # _wrap  #Temp array to hold data
        temp_del = [i for i in range(len(arr)) if arr[i] == 0]  # Pick out features not in this combo, and remove
        data = np.delete(data, temp_del, axis=1)
        data_train, data_test, target_train, target_test = train_test_split(data, target_np, test_size=0.35)

        if binning == 1:
            if bin_cnt <= 2:
                scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
                scores = cross_validate(clf, data_np, target_np, scoring=scorers, cv=5)
                score = scores['test_roc_auc'].mean()  # AUC
            else:
                scorers = {'Accuracy': 'accuracy'}
                scores = cross_validate(clf, data_np, target_np, scoring=scorers, cv=5)
                score = scores['test_Accuracy'].mean()  # Accuracy
            print('Random Forest Acc/AUC:', curr_idx, feat_arr, len(data[0]), score)
            if score > best_score:  # Compare performance and update sel_idx and best_score, if needed
                best_score = score
                sel_idx = copy.deepcopy(arr)

        if binning == 0:
            scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
            scores = cross_validate(clf, data, target_np, scoring=scorers, cv=5)
            score = np.asarray([math.sqrt(-x) for x in scores['test_Neg_MSE']]).mean()  # RMSE
            print('Random Forest Acc:', curr_idx, feat_arr, len(data[0]), score)
            if score < best_score:  # Compare performance and update sel_idx and best_score, if needed
                best_score = score
                sel_idx = copy.deepcopy(arr)

                # move to next feature index and recurse
        combo_ctr += 1
        curr_idx += 1
        feat_space_search(arr, curr_idx)  # Recurse till end of iteration for roll


# List Difference
def Diff(li1, li2):
    return list(set(li1) - set(li2))


# Hire / No Hire
def hire(x, median):
    if x > median:
        return 1
    else:
        return 0


def filter_cols(df, regex):
    rec = re.compile(regex)
    cols = list(df.columns)
    to_drop = [col for col in cols if rec.match(col)]
    new_df = df.drop(to_drop, axis=1)
    return new_df


def filter_cols_multi(df, lst):
    for regex in lst:
        df = filter_cols(df, regex)
    return df


##################################
# Load Data
##################################

train_data = pd.read_csv('Data/{}.csv'.format(train_filename))
test_data = pd.read_csv('Data/{}.csv'.format(test_filename))
test_data = test_data.fillna(test_data.mean())
labels = train_data[label]
target = train_data[target_var].fillna(0)
prot_group = train_data[prot_var]
high_perf = train_data["High_Performer"].dropna()
retained = train_data["Retained"]
test_labels = test_data[label]

exclusion_list = [r'.*_Time_.*', r'SJ_Most.*', r'SJ_Least.*', r'Scenario.*', r'hasPerf']
df = filter_cols_multi(train_data, exclusion_list)

df['SJ_Sum'] = df.filter(regex='SJ_Total.*').sum(axis=1)
df = filter_cols(df, r'SJ_Total.*')

features = df[list(df.columns)[9:]]

header = list(df.columns)[9:]
data_np = features.to_numpy()
target_np = target.to_numpy()

##################################
# Preprocess Data
##################################

if norm_target == 1:
    # Target normalization for continuous values
    target_np = scale(target_np)

if norm_features == 1:
    # Feature normalization for continuous values
    data_np = scale(data_np)

if binning == 1:
    # Discretize Target variable with KBinsDiscretizer
    enc = KBinsDiscretizer(n_bins=[bin_cnt], encode='ordinal',
                           strategy='quantile')  # Strategy here is important, quantile creating equal bins, but kmeans prob being more valid "clusters"
    target_np_bin = enc.fit_transform(target_np.reshape(-1, 1))

    # Get Bin min/max
    temp = [[] for x in range(bin_cnt + 1)]
    for i in range(len(target_np)):
        for j in range(bin_cnt):
            if target_np_bin[i] == j:
                temp[j].append(target_np[i])

    for j in range(bin_cnt):
        print('Bin', j, ':', min(temp[j]), max(temp[j]), len(temp[j]))
    print('\n')

    # Convert Target array back to correct shape
    target_np = np.ravel(target_np_bin)

##################################
# Over / Under Sampling
##################################
if undersample == 1:
    if us_type == 1:
        rus = RandomUnderSampler(replacement=True)
        data_np, target_np = rus.fit_resample(data_np, target_np)

    if us_type == 2:
        nm = NearMiss()
        data_np, target_np = nm.fit_resample(data_np, target_np)

    if us_type == 3:
        cc = ClusterCentroids()
        data_np, target_np = cc.fit_resample(data_np, target_np)

if oversampling == 1:
    if os_type == 1:
        ros = RandomOverSampler()
        data_np, target_np = ros.fit_resample(data_np, target_np)

    if os_type == 2:
        smote = SMOTE()
        data_np, target_np = smote.fit_resample(data_np, target_np)

    if os_type == 3:
        adasyn = ADASYN()
        data_np, target_np = adasyn.fit_resample(data_np, target_np)

##################################
# Grid Search
##################################

if grid_search == 1:
    if model_type == 'SVC':
        parameterGrid = hp_grid
        dt = DecisionTreeClassifier()
        clf = GridSearchCV(dt, parameterGrid, cv=5, n_jobs=-1)
        clf.fit(data_np, target_np)
        results = pd.DataFrame(clf.cv_results_)
        print(clf.best_estimator_)
        Model = clf.best_estimator_
    if model_type == 'RF':
        parameterGrid = hp_grid
        dt = RandomForestClassifier()
        clf = GridSearchCV(dt, parameterGrid, cv=5, n_jobs=-1)
        clf.fit(data_np, target_np)
        results = pd.DataFrame(clf.cv_results_)
        print(clf.best_estimator_)
        Model = clf.best_estimator_
    if model_type == 'GB':
        parameterGrid = hp_grid
        dt = GradientBoostingClassifier()
        clf = GridSearchCV(dt, parameterGrid, cv=5, n_jobs=-1)
        clf.fit(data_np, target_np)
        results = pd.DataFrame(clf.cv_results_)
        print(clf.best_estimator_)
        Model = clf.best_estimator_
    if model_type == 'ADA':
        parameterGrid = hp_grid
        dt = AdaBoostClassifier()
        clf = GridSearchCV(dt, parameterGrid, cv=5, n_jobs=-1)
        clf.fit(data_np, target_np)
        results = pd.DataFrame(clf.cv_results_)
        print(clf.best_estimator_)
        Model = clf.best_estimator_
    if model_type == 'MLP':
        parameterGrid = hp_grid
        dt = MLPClassifier()
        clf = GridSearchCV(dt, parameterGrid, cv=5, n_jobs=-1)
        clf.fit(data_np, target_np)
        results = pd.DataFrame(clf.cv_results_)
        print(clf.best_estimator_)
        Model = clf.best_estimator_
    if model_type == 'SVR':
        parameterGrid = hp_grid
        dt = SVC()
        clf = GridSearchCV(dt, parameterGrid, cv=5, n_jobs=-1)
        clf.fit(data_np, target_np)
        results = pd.DataFrame(clf.cv_results_)
        print(clf.best_estimator_)
        Model = clf.best_estimator_

if grid_search == 0:
    if model_type == 'RF':
        Model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                       criterion='gini', max_depth=20, max_features='auto',
                                       max_leaf_nodes=None, max_samples=None,
                                       min_impurity_decrease=0.0, min_impurity_split=None,
                                       min_samples_leaf=4, min_samples_split=4,
                                       min_weight_fraction_leaf=0.0, n_estimators=400,
                                       n_jobs=-1, oob_score=False, random_state=None,
                                       verbose=0, warm_start=False)
    if model_type == 'LR':
        Model = LogisticRegression(class_weight='balanced')

##################################
# Feature Selection
##################################

# Low Variance Filter
if lv_filter == 1:
    print('--LOW VARIANCE FILTER ON--', '\n')

    # LV Threshold
    sel = VarianceThreshold(threshold=0.5)  # Removes any feature with less than 20% variance
    fit_mod = sel.fit(data_np)
    fitted = sel.transform(data_np)
    sel_idx = fit_mod.get_support()

    # Get lists of selected and non-selected features (names and indexes)
    temp = []
    temp_idx = []
    temp_del = []
    for i in range(len(data_np[0])):
        if sel_idx[i] == 1:  # Selected Features get added to temp header
            temp.append(header[i + feat_start])
            temp_idx.append(i)
        else:  # Indexes of non-selected features get added to delete array
            temp_del.append(i)

    print('Selected:', temp)
    print('Removed:', Diff(header, temp))
    print('Features (total, selected):', len(data_np[0]), len(temp))
    print('\n')

    # Filter selected columns from original dataset
    header = header[0:feat_start]
    for field in temp:
        header.append(field)
    data_np = np.delete(data_np, temp_del, axis=1)  # Deletes non-selected features by index
    test_data = test_data[header]

if feat_select == 1:
    '''Three steps:
       1) Run Feature Selection
       2) Get lists of selected and non-selected features
       3) Filter columns from original dataset
       '''

    print('--FEATURE SELECTION ON--', '\n')

    # 1) Run Feature Selection
    if fs_type == 1:
        # Stepwise Recursive Backwards Feature removal
        rgr = Model
        sel = RFE(rgr, n_features_to_select=k_cnt, step=.1)
        print('Stepwise Recursive Backwards - {}: '.format(model_type))

        fit_mod = sel.fit(data_np, target_np)
        print(sel.ranking_)
        sel_idx = fit_mod.get_support()

    if fs_type == 2:
        # Wrapper Select via model
        if binning == 1:
            clf = Model
            sel = SelectFromModel(clf, prefit=False, threshold='mean',
                                  max_features=None)  # to select only based on max_features, set to integer value and set threshold=-np.inf
            print('Wrapper Select: ')
        if binning == 0:
            rgr = Model
            sel = SelectFromModel(rgr, prefit=False, threshold='mean', max_features=None)
            print('Wrapper Select: ')

        fit_mod = sel.fit(data_np, target_np)
        sel_idx = fit_mod.get_support()

    if fs_type == 3:
        if binning == 1:  # Only work if the Target is binned
            # Univariate Feature Selection - Chi-squared
            sel = SelectKBest(chi2, k=k_cnt)
            fit_mod = sel.fit(data_np,
                              target_np)  # will throw error if any negative values in features, so turn off feature normalization, or switch to mutual_info_classif
            print('Univariate Feature Selection - Chi2: ')
            sel_idx = fit_mod.get_support()

        if binning == 0:  # Only work if the Target is continuous
            # Univariate Feature Selection - Mutual Info Regression
            sel = SelectKBest(mutual_info_classif, k=k_cnt)
            fit_mod = sel.fit(data_np, target_np)
            print('Univariate Feature Selection - Mutual Info: ')
            sel_idx = fit_mod.get_support()

        # Print ranked variables out sorted
        temp = []
        scores = fit_mod.scores_
        for i in range(feat_start, len(header)):
            temp.append([header[i], float(scores[i - feat_start])])

        print('Ranked Features')
        temp_sort = sorted(temp, key=itemgetter(1), reverse=True)
        for i in range(len(temp_sort)):
            print(i, temp_sort[i][0], ':', temp_sort[i][1])
        print('\n')

    if fs_type == 4:
        # Full-blown Wrapper Select (from any kind of ML model)
        if binning == 1:  # Only work if the Target is binned
            start_ts = time.time()
            sel_idx = []  # Empty array to hold optimal selected feature set
            best_score = 0  # For classification compare Accuracy or AUC, higher is better, so start with 0
            feat_cnt = len(data_np[0])
            # Create Wrapper model
            clf = Model  # This could be any kind of classifier model

        if binning == 0:  # Only work if the Target is continuous
            start_ts = time.time()
            sel_idx = []  # Empty array to hold optimal selected feature set
            best_score = sys.float_info.max  # For regression compare RMSE, lower is better
            feat_cnt = len(data_np[0])
            # Create Wrapper model
            clf = Model  # This could be any kind of regressor model

        # Loop thru feature sets
        roll_idx = 0
        combo_ctr = 0
        feat_arr = [0 for col in range(feat_cnt)]  # Initialize feature array
        for idx in range(feat_cnt):
            roll_idx = idx
            feat_space_search(feat_arr, idx)  # Recurse
            feat_arr = [0 for col in range(feat_cnt)]  # Reset feature array after each iteration

        print('# of Feature Combos Tested:', combo_ctr)
        print(best_score, sel_idx, len(data_np[0]))
        print("Wrapper Feat Sel Runtime:", time.time() - start_ts)

    # 2) Get lists of selected and non-selected features (names and indexes)
    temp = []
    temp_idx = []
    temp_del = []
    for i in range(len(data_np[0])):
        if sel_idx[i] == 1:  # Selected Features get added to temp header
            temp.append(header[i + feat_start])
            temp_idx.append(i)
        else:  # Indexes of non-selected features get added to delete array
            temp_del.append(i)
    print('Selected', temp)
    print('Features (total/selected):', len(data_np[0]), len(temp))
    print('\n')

    # 3) Filter selected columns from original dataset
    header = header[0:feat_start]
    for field in temp:
        header.append(field)
    data_np = np.delete(data_np, temp_del, axis=1)  # Deletes non-selected features by index)

##################################
# Train Models
##################################

print('--ML Model Output--', '\n')

# Test/Train split
data_train, data_test, target_train, target_test = train_test_split(data_np, target_np, test_size=0.35)

# Classifiers
if binning == 0 and cross_val == 0:
    clf = Model
    clf.fit(data_train, target_train)

    scores_ACC = clf.score(data_test, target_test)
    print('{} Acc:'.format(model_type), scores_ACC)
    scores_AUC = metrics.roc_auc_score(target_test, clf.predict_proba(data_test)[:, 1])
    print('{} AUC:'.format(model_type), scores_AUC)

# Cross-Validation Classifiers
if binning == 0 and cross_val == 1:
    # Setup cross-validation classification scorers
    scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}

    # SciKit SVM - Cross Val
    start_ts = time.time()
    clf = Model
    scores = cross_validate(estimator=clf, X=data_np, y=target_np, scoring=scorers, cv=5)

    scores_Acc = scores['test_Accuracy']
    print("SVM Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))
    scores_AUC = scores['test_roc_auc']  # Only works with binary classes, not multiclass
    print("AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))
    print("CV Runtime:", time.time() - start_ts, '\n')

##################################
# SIOP Testing Formula
##################################

clf.fit(data_train, target_train)
#feature_importance = pd.concat([pd.Series(header), pd.Series(clf.feature_importances_)], axis=1)
#feature_importance = feature_importance.sort_values(by=1, ascending=False)
pred = clf.predict(data_train)
print(metrics.confusion_matrix(target_train, pred, normalize='true'))

coef = pd.DataFrame({'feature': header, 'coef': clf.coef_[0], 'exp': np.exp(clf.coef_[0]),
                    'abs_value': np.abs(clf.coef_[0])} )

print(coef.sort_values(by='abs_value', ascending=False))


pred = clf.predict(data_test)
print(metrics.confusion_matrix(target_test, pred, normalize='true'))

# train_results = pd.DataFrame(clf.predict_proba(data_np))
# train_pred = pd.concat([labels, high_perf, retained, prot_group, train_results], axis=1)
# hybrids = []
# for i2 in range(0, len(train_pred)):
#     if train_pred.iloc[i2]['High_Performer'] == 1 and train_pred.iloc[i2]['Retained'] == 1:
#         hybrids.append(1)
#     else:
#         hybrids.append(0)
# train_pred.insert(3, 'Hybrid', hybrids)
# median = train_pred[1].median()
# train_pred["Hire"] = train_pred[1].apply(lambda x: hire(x, median))
#
# # Retained Classification
# retained_classification = []
# for i in range(0, len(train_pred)):
#     retained = train_pred.iloc[i]["Retained"]
#     h = train_pred.iloc[i]["Hire"]
#     if retained == 1 and h == 1:
#         retained_classification.append('TP')
#     if retained == 0 and h == 0:
#         retained_classification.append('TN')
#     if retained == 1 and h == 0:
#         retained_classification.append('FN')
#     if retained == 0 and h == 1:
#         retained_classification.append('FP')
# train_pred['Retained_Classification'] = retained_classification
# R_TP = len(train_pred[train_pred['Retained_Classification'] == 'TP'])
# R_TN = len(train_pred[train_pred['Retained_Classification'] == 'TN'])
# R_FP = len(train_pred[train_pred['Retained_Classification'] == 'FP'])
# R_FN = len(train_pred[train_pred['Retained_Classification'] == 'FN'])
# R_Hire_Percentage = R_TP / (R_TP + R_FN)
#
# # High Performance Classification
# hp_train_pred = train_pred[train_pred['High_Performer'].notnull()]
# high_performance_classification = []
# for i in range(0, len(hp_train_pred)):
#     high_performer = hp_train_pred.iloc[i]["High_Performer"]
#     h = hp_train_pred.iloc[i]["Hire"]
#     if high_performer == 1 and h == 1:
#         high_performance_classification.append('TP')
#     if high_performer == 0 and h == 0:
#         high_performance_classification.append('TN')
#     if high_performer == 1 and h == 0:
#         high_performance_classification.append('FN')
#     if high_performer == 0 and h == 1:
#         high_performance_classification.append('FP')
# hp_train_pred['High_Perf_Classification'] = high_performance_classification
# HP_TP = len(hp_train_pred[hp_train_pred['High_Perf_Classification'] == 'TP'])
# HP_TN = len(hp_train_pred[hp_train_pred['High_Perf_Classification'] == 'TN'])
# HP_FP = len(hp_train_pred[hp_train_pred['High_Perf_Classification'] == 'FP'])
# HP_FN = len(hp_train_pred[hp_train_pred['High_Perf_Classification'] == 'FN'])
# HP_Hire_Percentage = HP_TP / (HP_TP + HP_FN)
#
# # Hybrid Classification
# hybrid_classification = []
# for i in range(0, len(hp_train_pred)):
#     high_performer = hp_train_pred.iloc[i]["High_Performer"]
#     retained = hp_train_pred.iloc[i]["Retained"]
#     if high_performer == 1 and retained == 1:
#         hybrid = 1
#     else:
#         hybrid = 0
#     h = hp_train_pred.iloc[i]["Hire"]
#     if hybrid == 1 and h == 1:
#         hybrid_classification.append('TP')
#     if hybrid == 0 and h == 0:
#         hybrid_classification.append('TN')
#     if hybrid == 1 and h == 0:
#         hybrid_classification.append('FN')
#     if hybrid == 0 and h == 1:
#         hybrid_classification.append('FP')
# hp_train_pred['Hybrid_Classification'] = hybrid_classification
# H_TP = len(hp_train_pred[hp_train_pred['Hybrid_Classification'] == 'TP'])
# H_TN = len(hp_train_pred[hp_train_pred['Hybrid_Classification'] == 'TN'])
# H_FP = len(hp_train_pred[hp_train_pred['Hybrid_Classification'] == 'FP'])
# H_FN = len(hp_train_pred[hp_train_pred['Hybrid_Classification'] == 'FN'])
# H_Hire_Percentage = H_TP / (H_TP + H_FN)
#
# # Adverse Impact Ratio
# prot_hired = len(train_pred.loc[(train_pred['Hire'] == 1) & (train_pred['Protected_Group'] == 1)])
# prot_not_hired = len(train_pred.loc[(train_pred['Hire'] == 0) & (train_pred['Protected_Group'] == 1)])
# non_prot_hired = len(train_pred.loc[(train_pred['Hire'] == 1) & (train_pred['Protected_Group'] == 0)])
# non_prot_not_hired = len(train_pred.loc[(train_pred['Hire'] == 0) & (train_pred['Protected_Group'] == 0)])
# adverse_impact = (prot_hired / (prot_not_hired + prot_hired)) / (non_prot_hired / (non_prot_not_hired + non_prot_hired))
#
# # Final Score
# score = ((HP_Hire_Percentage * 25) + (R_Hire_Percentage * 25) + (H_Hire_Percentage * 50)) - \
#         (abs(1 - adverse_impact) * 100)
# print("High Performers Hired: {0:.2f},"
#       " Retained Hired: {1:.2f},"
#       " Hybrid Hired: {2:.2f}"
#       " Adverse Impact: {3:.2f},"
#       " Score: {4:.2f}%".format(HP_Hire_Percentage,
#                                 R_Hire_Percentage,
#                                 H_Hire_Percentage,
#                                 adverse_impact,
#                                 score))
#
# ##################################
# # Predict Using Model
# ##################################
#
# print('--ML Model Prediction--', '\n')
#
# clf = Model
# clf.fit(data_np, target_np)
# results = pd.DataFrame(clf.predict_proba(test_data[header]))
# final_predictions = pd.concat([test_labels, results], axis=1)
# median = final_predictions[1].median()
# final_predictions["Hire"] = final_predictions[1].apply(lambda x: hire(x, median))

##################################
# Performance & Predictions to SQL
##################################

# # Model Configuration
# mc = {'modelID': modelid,
#       'modelType': model_type,
#       'modelTarget': target_var,
#       'undersampleFlag': undersample,
#       'undersampleType': us_type,
#       'oversampleFlag': oversampling,
#       'oversampleType': os_type,
#       'lvFilterFlag': lv_filter,
#       'featureSelect': feat_select,
#       'fsType': fs_type}
# model_config = pd.DataFrame(mc, index=[0])
# model_config.to_sql('modelConfiguration', engine, if_exists='append', index=False, schema='ax')
#
# # Model Score
# fs = {'modelID': modelid,
#       'Percentage_of_true_top_performers_hired': HP_Hire_Percentage,
#       'Percentage_of_true_retained_hired': R_Hire_Percentage,
#       'Percentage_of_true_retained_top_performers_hired': H_Hire_Percentage,
#       'Adverse_impact_ratio': adverse_impact,
#       'Final_score': score}
# final_scores = pd.DataFrame(fs, index=[0])
# final_scores.to_sql('modelPerformance', engine, if_exists='append', index=False, schema='ax')
#
# # Model Predictions
# final_predictions.insert(0, "modelID", modelid)
# final_predictions.to_sql('modelPredictions', engine, if_exists='append', index=False, schema='ax')
#
# # Classification Details
# hp_train_pred.to_sql('highPerformerClassifications', engine, if_exists='append', index=False, schema='ax')
# train_pred.to_sql('retentionClassifications', engine, if_exists='append', index=False, schema='ax')
