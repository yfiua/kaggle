import numpy as np
import pandas as pd

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

# prepare data
y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1).values

id_test = df_test['ID']
X_test = df_test.drop(['ID'], axis=1).values

# length of dataset
len_train = len(X_train)
len_test  = len(X_test)
n_col = X_train.shape[1]

# missing values
missing_values = [ -999999.0, 9999999999.0, 117310.979016494 ]
for v in missing_values:
    X_train[X_train == v] = np.nan
    X_test[X_test == v] = np.nan

# find common values between columns
X = np.concatenate((X_train, X_test))
common = np.zeros([n_col, n_col])
for i in range(n_col):
    col = X[:, i]
    for j in range(i+1, n_col):
        common[i, j] = sum(np.logical_and(col == X[:, j], (col != 0), np.logical_not(np.isnan(col))))
        common[j, i] = common[i, j]

## output numbers of common values between columns
c = df_train.columns
f = open('n_common.csv', 'w')
f.write('0')
for i in range(n_col):
    f.write(',(' + str(i) + ')' + c[i+1])

for i in range(n_col):
    f.write('\n(' + str(i) + ')' + c[i+1])
    for j in range(n_col):
        f.write(',' + str(common[i, j]))

f.close()

print 'Completed!'
