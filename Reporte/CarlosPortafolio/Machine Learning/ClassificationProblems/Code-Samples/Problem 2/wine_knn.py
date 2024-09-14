#------------------------------------------------------------------------------------------------------------------
#   Evaluation the KNN classifier with hyperparameter selection for the Wine data set
#------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Load data
wine = datasets.load_wine()
x = wine.data
y = wine.target
features = wine.feature_names
n_features = len(features)

################################################################################
# Evaluation
################################################################################
print("----- Model evaluation -----")
kf = StratifiedKFold(n_splits=5, shuffle = True)

cv_y_test = []
cv_y_pred = []

for train_index, test_index in kf.split(x, y):
    
    x_train = x[train_index, :]
    y_train = y[train_index]

    x_test = x[test_index, :]
    y_test = y[test_index]

    parameters = {'n_neighbors': np.arange(1, 100)}
    clf_cv = GridSearchCV(KNeighborsClassifier(), parameters, cv = 5)
    clf_cv.fit(x_train, y_train)

    y_pred = clf_cv.predict(x_test)
    
    cv_y_test.append(y_test)
    cv_y_pred.append(y_pred)

print(classification_report(np.concatenate(cv_y_test), np.concatenate(cv_y_pred)))

################################################################################
# Evaluation with cross_val_predict
################################################################################
print("----- Model evaluation with cross_val_predict -----")

clf = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': np.arange(1, 100)}, cv = 5)
y_pred = cross_val_predict(clf, x, y, cv = 5)

print(classification_report(y, y_pred))

################################################################################
# Production model
################################################################################
print("----- Production model -----")

clf = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': np.arange(1, 100)}, cv = 5)
clf.fit(x, y)

print(clf.best_estimator_)

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------