from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

# load data
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

# remove constant columns
remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = df_train.columns
for i in range(len(c)-1):
    v = df_train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,df_train[c[j]].values):
            remove.append(c[j])

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)


y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values

id_test = df_test['ID']
X_test = df_test.drop(['ID'], axis=1).values

# length of dataset
len_train = len(X_train)
len_test  = len(X_test)

# normal case
X_train_norm = [ X_train[i] for i in range(len_train) if X_train[i, -1] != 117310.979016494 ]
y_train_norm = [ y_train[i] for i in range(len_train) if X_train[i, -1] != 117310.979016494 ]
X_test_norm  = [ X_test[i]  for i in range(len_test)  if X_test[i, -1]  != 117310.979016494 ]
id_test_norm = [ id_test[i] for i in range(len_test)  if X_test[i, -1]  != 117310.979016494 ]

X_train_norm = np.array(X_train_norm)

# special case
X_test_spec  = [ X_test[i]  for i in range(len_test)  if X_test[i, -1]  == 117310.979016494 ]
id_test_spec = [ id_test[i] for i in range(len_test)  if X_test[i, -1]  == 117310.979016494 ]

X_train_spec = np.array(X_train)[:,0:-1]
y_train_spec = y_train
X_test_spec  = np.array(X_test)[:,0:-1]

# classifier
clf_norm = xgb.XGBClassifier(max_depth=7, n_estimators=300, learning_rate=0.05, nthread=4)
clf_spec = xgb.XGBClassifier(max_depth=7, n_estimators=300, learning_rate=0.05, nthread=4)

#scores = cross_validation.cross_val_score(clf_norm, X_train_norm, y_train_norm, scoring='roc_auc', cv=5)
#print(scores.mean())

X_train_norm, X_eval_norm, y_train_norm, y_eval_norm = train_test_split(X_train_norm, y_train_norm, test_size=0.33, random_state=42)
X_train_spec, X_eval_spec, y_train_spec, y_eval_spec = train_test_split(X_train_spec, y_train_spec, test_size=0.33, random_state=42)

# fitting
clf_norm.fit(X_train_norm, y_train_norm, early_stopping_rounds=10, eval_metric="auc", eval_set=[(X_eval_norm, y_eval_norm)])
clf_spec.fit(X_train_spec, y_train_spec, early_stopping_rounds=10, eval_metric="auc", eval_set=[(X_eval_norm, y_eval_norm)])

# predicting
y_pred_norm = clf_norm.predict_proba(X_test)
y_pred_spec = clf_spec.predict_proba(X_test_spec)

# ensemble
y_pred = [ y_pred_norm[i] if X_test[i, -1] != 117310.979016494 else y_pred_spec[i] for i in range(len_test) ]
y_pred = np.array(y_pred)

submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)
