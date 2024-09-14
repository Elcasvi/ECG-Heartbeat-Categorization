#------------------------------------------------------------------------------------------------------------------
#   Multilayer perceptrom for the Iris dataset (Keras + TensorFlow)
#------------------------------------------------------------------------------------------------------------------

import numpy as np

from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Import IRIS dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target

targets = iris.target_names
n_clases = len(targets)

features = iris.feature_names
n_features = len(features)

# Create output variables from original labels. This is required only in multiclass problems.
output_y = to_categorical(y)   
print(output_y)

# Define MLP model
clf = Sequential()
clf.add(Dense(10, input_dim=n_features, activation='relu'))
clf.add(Dense(10, activation='relu'))
clf.add(Dense(3, activation='softmax')) # for 2-class problems, use clf.add(Dense(1, activation='sigmoid'))

# Compile model
clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
# Fit model
clf.fit(x, output_y, epochs=150, batch_size=5)

# Predict class of a new observation
prob = clf.predict( [[1.,2.,3.,4.]] )
print("Probablities", prob)
print("Predicted class", np.argmax(prob, axis=-1)) # For 2-class problems, use (prob > 0.5).astype("int32")

# Evaluate model
kf = StratifiedKFold(n_splits=5, shuffle = True)

cv_y_test = []
cv_y_pred = []

for train_index, test_index in kf.split(x, y):

    x_train = x[train_index, :]
    y_train = y[train_index]
    y_train_categorical = to_categorical(y_train)

    x_test = x[test_index, :]
    y_test = y[test_index]
    y_test_categorical = to_categorical(y_test)

    # Training phase
    clf_cv = Sequential()
    clf_cv.add(Dense(10, input_dim=n_features, activation='relu'))
    clf_cv.add(Dense(10, activation='relu'))
    clf_cv.add(Dense(3, activation='softmax'))
    clf_cv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    clf_cv.fit(x_train, y_train_categorical, validation_data= (x_test, y_test_categorical), epochs=150, batch_size=5)    

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]
    y_pred = np.argmax(clf_cv.predict(x_test), axis=-1)  

    cv_y_test.append(y_test)
    cv_y_pred.append(y_pred)


print(classification_report(np.concatenate(cv_y_test), np.concatenate(cv_y_pred)))

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------