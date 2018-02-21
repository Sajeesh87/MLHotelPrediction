#import libraries
import pandas as pd

#importing dataset
dataset_Destination = pd.read_csv('destinations.csv')
dataset_sample_submission = pd.read_csv('sample_submission.csv')
dataset_train = pd.read_csv('test.csv')
dataset_test = pd.read_csv('train.csv')

#remove colums
for col in dataset_test:
    if(col in dataset_train):
        pass
    else:
        del dataset_test[col]
for col in dataset_train:
    if(col in dataset_test):
        pass
    else:
        del dataset_train[col]

#split dependent and indepandent data
X = dataset_train.iloc[:, :]
y = dataset_train.iloc[:, 17].values
X_test = dataset_test.iloc[:, :]
y_test = dataset_test.iloc[:, 16].values

#categorical data
from sklearn.preprocessing import LabelEncoder#,OneHotEncoder
LEX_obj = LabelEncoder()
for col in X.columns:
    X[col] = LEX_obj.fit_transform(X[col].astype(str))
for col in X_test.columns:
    X_test[col] = LEX_obj.fit_transform(X_test[col].astype(str))
    
X = X.values
X_test = X_test.values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#missing data
from sklearn.preprocessing import Imputer
imputer_obj = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imputer_obj = imputer_obj.fit(X)
X = imputer_obj.transform(X)

# prepare configuration for cross validation test harness
seed = 7

#import algo
import sklearn.metrics as met
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# prepare models
models = []
models.append(('NB', GaussianNB()))
models.append(('CART', DecisionTreeClassifier()))

# evaluate each model in turn
for name,model in models:
    m = model
    m.fit(X,y)
    y_pred = m.predict(X_test)
    print(met.accuracy_score(y_test,y_pred))