import pandas as pd
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split

fname = 'Data/multiclass.csv'
model = RandomForestClassifier(n_estimators=200, max_depth=50, min_samples_leaf=2, min_samples_split=2)

df = pd.read_csv(fname)

features = list(df.columns)[9:-3]
features.remove('hasPerf')

target_np = df['target'].to_numpy()
data_np = df[features].to_numpy()

print('--ML Model Output--', '\n')

# Test/Train split
data_train, data_test, target_train, target_test = train_test_split(data_np, target_np, test_size=0.35)

scorers = {'Accuracy': 'accuracy'}

# SciKit SVM - Cross Val
start_ts = time.time()
clf = model
scores = cross_validate(estimator=clf, X=data_train, y=target_train, scoring=scorers, cv=5)

scores_Acc = scores['test_Accuracy']
print("SVM Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))
# scores_AUC = scores['test_roc_auc']  # Only works with binary classes, not multiclass
# print("AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))
print("CV Runtime:", time.time() - start_ts, '\n')

clf.fit(data_train, target_train)
pred = clf.predict(data_test)

print(metrics.confusion_matrix(target_test, pred, normalize='true'))
print(metrics.confusion_matrix(target_test, pred))

