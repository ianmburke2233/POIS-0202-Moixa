import pandas as pd
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split

fname = 'Data/multiclass.csv'
dev_fname = 'Data/multiclassDev.csv'
outname = '2020-09-18 Random Forest (Protected Group, High Performers Only)'
model = RandomForestClassifier(n_estimators=200, max_depth=50, min_samples_leaf=2, min_samples_split=2)
population = 1  # 1 - Full Population, 2 - Protected Group Only, 3 - Non-Protected Group Only
target = 2  # 1 - Hybrid Target, 2 - High Performer, 3 - Retained

df = pd.read_csv(fname)
df = df[df['Protected_Group'].isna() == False]

if population == 2:
    df = df[df['Protected_Group'] == 1]
if population == 3:
    df = df[df['Protected_Group'] == 0]

dev_df = pd.read_csv(dev_fname)

ids = dev_df['UNIQUE_ID']

features = list(df.columns)[10:-3]
features.remove('hasPerf')

if target == 1:
    target_np = df['Target'].to_numpy()
if target == 2:
    target_np = df['Filled_High_Performer'].to_numpy()
if target == 3:
    target_np = df['Retained'].to_numpy()

data_np = df[features].to_numpy()

print('--ML Model Output--', '\n')


def hire(mid, x):
    if x > mid:
        decision = 1
    else:
        decision = 0
    return decision


clf = RandomForestClassifier(max_depth=3, class_weight='balanced', n_estimators=100, random_state=20200314)

y = df['Protected_Group']

X_train, X_test, y_train, y_test = train_test_split(data_np, y, test_size=0.35)
scorers = {'Accuracy': 'accuracy'}
scores = cross_validate(estimator=clf, X=X_train, y=y_train, scoring=scorers, cv=5)


scores_Acc = scores['test_Accuracy']
print("SVM Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))

clf.fit(data_np, y)
data_dev = dev_df[features]
dev_pred = clf.predict(data_dev)
dev_pred_prob = clf.predict_proba(data_dev)
dev_predicted_class = pd.DataFrame(dev_pred, columns=['Predicted Class'])
results = dev_predicted_class.join(pd.DataFrame(dev_pred_prob, columns=['Class 1 Prob', 'Class 2 Prob']))
results = results.join(ids).sort_values(by='Class 2 Prob')

median = results['Class 2 Prob'].median()
results['Protected'] = results['Class 2 Prob'].apply(lambda x: hire(median, x))

results.to_csv('Data/Predicted Protected Status.csv')