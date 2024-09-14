#------------------------------------------------------------------------------------------------------------------
#   Neural network for the Wine dataset
#------------------------------------------------------------------------------------------------------------------
import numpy as np

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report

# Load data
wine = datasets.load_wine()
x = wine.data
y = wine.target
features = wine.feature_names
n_features = len(features)

#------------------------------------------------------------------------------------------------------------------
# Train model with all observations (two layers of 100 neurons)
#------------------------------------------------------------------------------------------------------------------

clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10000)  
clf.fit(x, y)

#------------------------------------------------------------------------------------------------------------------
# Evaluate model (two layers of 100 neurons)
#------------------------------------------------------------------------------------------------------------------

y_pred = cross_val_predict(MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=10000), x, y)
print(classification_report(y, y_pred))

#------------------------------------------------------------------------------------------------------------------
# Optimal number of layers and neurons
#------------------------------------------------------------------------------------------------------------------

num_layers = np.arange(1, 20, 5)
num_neurons = np.arange(10, 110, 20)

layers = []

for l in num_layers:
    for n in num_neurons:
        layers.append(l*[n])

clf = GridSearchCV(MLPClassifier(max_iter=10000), {'hidden_layer_sizes': layers}, cv = 5)
clf.fit(x, y)
print(clf.best_estimator_)

#------------------------------------------------------------------------------------------------------------------
# Evaluate model finding the optimal number of layers and neurons
#------------------------------------------------------------------------------------------------------------------

clf = GridSearchCV(MLPClassifier(max_iter=10000), {'hidden_layer_sizes': layers}, cv = 5)
y_pred = cross_val_predict(clf, x, y, cv = 5)
print(classification_report(y, y_pred))

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------