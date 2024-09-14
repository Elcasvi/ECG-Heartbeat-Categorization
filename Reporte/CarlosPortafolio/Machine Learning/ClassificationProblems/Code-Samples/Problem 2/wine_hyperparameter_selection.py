#------------------------------------------------------------------------------------------------------------------
#   Hyperparameter selection for some classification models fitted with the Wine dataset
#------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
wine = datasets.load_wine()
x = wine.data
y = wine.target
features = wine.feature_names
n_features = len(features)

#------------------------------------------------------------------------------------------------------------------
# K parameter of the KNN classifier
#------------------------------------------------------------------------------------------------------------------

print("----- KNN classifier - K parameter -----")

kk = np.arange(1,140)

acc = []

for k in kk:
    print('---- k =', k)
    
    acc_cv = []

    kf = StratifiedKFold(n_splits=5, shuffle = True)

    for train_index, test_index in kf.split(x, y):
    
        # Training phase
        x_train = x[train_index, :]
        y_train = y[train_index]     

        clf_cv = KNeighborsClassifier(n_neighbors=k)             

        clf_cv.fit(x_train, y_train)

        # Test phase
        x_test = x[test_index, :]
        y_test = y[test_index]
        y_pred = clf_cv.predict(x_test)    
        
        acc_i = accuracy_score(y_test, y_pred)
        acc_cv.append(acc_i)    

    acc_hyp = np.average(acc_cv)
    acc.append(acc_hyp)
    
    print('ACC:', acc_hyp)

opt_index = np.argmax(acc)
opt_hyperparameter = kk[opt_index]
print("Optimal k: ", opt_hyperparameter)

plt.plot(kk, acc)
plt.xlabel("k")
plt.ylabel("Accuracy")

plt.show()

# Fit model with optimal hyperparameters
clf = KNeighborsClassifier(n_neighbors=opt_hyperparameter)   
clf.fit(x, y)

#------------------------------------------------------------------------------------------------------------------
# Regularization parameter of the linear SVM classifier
#------------------------------------------------------------------------------------------------------------------

print("----- SVM classifier - Regularization parameter -----")

cc = np.logspace(-3, 1, 100)

acc = []

for c in cc:
    print('---- C =', c)
    
    acc_cv = []

    kf = StratifiedKFold(n_splits=5, shuffle = True)

    for train_index, test_index in kf.split(x, y):
    
        # Training phase
        x_train = x[train_index, :]
        y_train = y[train_index]     

        clf_cv = SVC(C=c, kernel = 'linear')            

        clf_cv.fit(x_train, y_train)

        # Test phase
        x_test = x[test_index, :]
        y_test = y[test_index]
        y_pred = clf_cv.predict(x_test)    
        
        acc_i = accuracy_score(y_test, y_pred)
        acc_cv.append(acc_i)    

    acc_hyp = np.average(acc_cv)
    acc.append(acc_hyp)
    
    print('ACC:', acc_hyp)

opt_index = np.argmax(acc)
opt_hyperparameter = cc[opt_index]
print("Optimal C: ", opt_hyperparameter)

plt.plot(cc, acc)
plt.xscale('log')
plt.xlabel("c")
plt.ylabel("Accuracy")

plt.show()

# Fit model with optimal optimal hyperparameters
clf = SVC(C=opt_hyperparameter, kernel = 'linear')
clf.fit(x, y)

#------------------------------------------------------------------------------------------------------------------
# Smoothing parameter of the RB-SVM classifier
#------------------------------------------------------------------------------------------------------------------

print("----- RB-SVM classifier - Smoothing parameter -----")

gg = np.logspace(-5, -1, 100)

acc = []

for g in gg:
    print('---- gamma =', g)
    
    acc_cv = []

    kf = StratifiedKFold(n_splits=5, shuffle = True)

    for train_index, test_index in kf.split(x, y):
    
        # Training phase
        x_train = x[train_index, :]
        y_train = y[train_index]     

        clf_cv = SVC(kernel ='rbf', gamma = g)         

        clf_cv.fit(x_train, y_train)

        # Test phase
        x_test = x[test_index, :]
        y_test = y[test_index]
        y_pred = clf_cv.predict(x_test)    
        
        acc_i = accuracy_score(y_test, y_pred)
        acc_cv.append(acc_i)    

    acc_hyp = np.average(acc_cv)
    acc.append(acc_hyp)
    
    print('ACC:', acc_hyp)

opt_index = np.argmax(acc)
opt_hyperparameter = gg[opt_index]
print("Optimal gamma: ", opt_hyperparameter)

plt.plot(gg, acc)
plt.xscale('log')
plt.xlabel("gamma")
plt.ylabel("Accuracy")

plt.show()

# Fit model with optimal optimal hyperparameters
clf = SVC(C=opt_hyperparameter, kernel = 'linear')
clf.fit(x, y)

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------