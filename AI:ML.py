#import libraries
import pandas as pd

#importing dataset
dataset_Destination = pd.read_csv('destinations.csv')
dataset_sample_submission = pd.read_csv('sample_submission.csv')
dataset_train = pd.read_csv('test.csv')
dataset_test = pd.read_csv('train.csv')

#split dependent and indepandent data
X = dataset_train.iloc[:, :17].values
y = dataset_train.iloc[:, 17:].values
X_test = dataset_test.iloc[:, :17].values
y_test = dataset_test.iloc[:, 17:].values

#categorical data
from sklearn.preprocessing import LabelEncoder
LEX_obj = LabelEncoder()
X[:,1] = LEX_obj.fit_transform(X[:,1])
X[:,12] = LEX_obj.fit_transform(X[:,12].astype(str))
X[:,13] = LEX_obj.fit_transform(X[:,13].astype(str))
''' no need to split categorical data hence commented
OHE_obj = OneHotEncoder(categorical_features=[0])
X = OHE_obj.fit_transform(X).toarray()
LE_obj = LabelEncoder()
y = LE_obj.fit_transform(y)
'''

#missing data
from sklearn.preprocessing import Imputer
imputer_obj = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imputer_obj = imputer_obj.fit(X)
X = imputer_obj.transform(X)

# prepare configuration for cross validation test harness
seed = 7

#import algo
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
# prepare models
models = []
#models.append(('NB', GaussianNB()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    print(name)
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()