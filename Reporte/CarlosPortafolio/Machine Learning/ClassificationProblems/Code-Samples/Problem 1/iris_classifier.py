#------------------------------------------------------------------------------------------------------------------
#   Iris dataset classifier
#------------------------------------------------------------------------------------------------------------------

import numpy as np

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm

#------------------------------------------------------------------------------------------------------------------
# Load dataset
#------------------------------------------------------------------------------------------------------------------

# Import IRIS dataset
iris = datasets.load_iris()

x = iris.data
print(x)
y = iris.target
print(y)
targets = iris.target_names
print(targets)
n_targets = len(targets)
features = iris.feature_names
n_features = len(features)

# Plot pairs of variables
cmap = cm.get_cmap('viridis')
fig, axs = plt.subplots(n_features, n_features)

for i in range(n_features):
    for j in range(n_features):
        if i != j:
            for k in range(n_targets):
                axs[i, j].scatter(x[y==k,j], x[y==k,i], label = targets[k], color = cmap(k/(n_targets-1)), 
                                  alpha=0.8, edgecolor='k')
        else:
            axs[i, j].text(0.5, 0.5, features[i], horizontalalignment='center', 
                           verticalalignment='center', style='italic', fontsize=14)
            axs[i, j].xaxis.set_visible(False)
            axs[i, j].yaxis.set_visible(False)
           

axs[n_features//2 - 1, n_features-1].legend(bbox_to_anchor=(1.5, 0.1))
plt.subplots_adjust(right=0.90)
plt.show()

#------------------------------------------------------------------------------------------------------------------
# Model training (with the complete dataset)
#------------------------------------------------------------------------------------------------------------------

# Train SVM classifier with all available observations
clf = SVC(kernel = 'linear')
clf.fit(x, y)

# Predict three new observations
print("New evaluations", clf.predict( [[4.4, 2.9, 1.4, 0.2], [0.4,0.1,0.1,0.4], [7.7, 3.0, 6.1, 2.2]]))

#------------------------------------------------------------------------------------------------------------------
# Model evaluation 
#------------------------------------------------------------------------------------------------------------------

# k-fold cross-validation
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle = True)

acc = 0
recall = np.array([0., 0., 0.])
precision = np.array([0., 0., 0.])

cv_y_test = []
cv_y_pred = []

for train_index, test_index in kf.split(x, y):
    
    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]

    clf_cv = SVC(kernel = 'linear')
    clf_cv.fit(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]    
    y_pred = clf_cv.predict(x_test)

    # Concatenate results of evaluation
    cv_y_test.append(y_test)
    cv_y_pred.append(y_pred)

    # Model performance    
    print(classification_report(y_test, y_pred))

    # Confussion matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix\n', cm)    

    # Performance scores
    acc += accuracy_score(y_test, y_pred)
    recall += recall_score(y_test, y_pred, average = None)
    precision += precision_score(y_test, y_pred, average = None)

# Print average performance
acc = acc/n_folds
print('Acc: ', acc)

precision = precision/n_folds
print('Precision: ', precision)

recall = recall/n_folds
print('Recall: ', recall)

# Total confussion matrix
cm = confusion_matrix(np.concatenate(cv_y_test), np.concatenate(cv_y_pred))
print('Confusion matrix\n', cm)    

# Model performance
print(classification_report(np.concatenate(cv_y_test), np.concatenate(cv_y_pred)))

#------------------------------------------------------------------------------------------------------------------
# Model evaluation using direct functions
#------------------------------------------------------------------------------------------------------------------

# k-fold cross-validation using cross_validate
n_folds = 5
clf_cv = SVC(kernel = 'linear')
cv_results = cross_validate(clf_cv, x, y, cv=n_folds, scoring = ('accuracy', 'recall_micro', 'precision_micro'))

print('Acc: ', cv_results['test_accuracy'].sum()/n_folds)
print('Recall: ', cv_results['test_recall_micro'].sum()/n_folds)
print('Precision: ', cv_results['test_precision_micro'].sum()/n_folds)

# k-fold cross-validation using cross_val_predict
n_folds = 5
clf_cv = SVC(kernel = 'linear')
y_pred = cross_val_predict(clf_cv, x, y, cv=n_folds)
print(classification_report(y, y_pred))

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------