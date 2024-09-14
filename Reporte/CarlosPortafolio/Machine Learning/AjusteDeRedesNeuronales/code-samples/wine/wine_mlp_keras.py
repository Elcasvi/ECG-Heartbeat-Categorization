#------------------------------------------------------------------------------------------------------------------
#   Multilayer perceptron for the Wine dataset (Keras + TensorFlow)
#------------------------------------------------------------------------------------------------------------------

import numpy as np

from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Load data
wine = datasets.load_wine()
x = wine.data
y = wine.target
features = wine.feature_names
n_features = len(features)

output_y = to_categorical(y)   

# Define MLP model
clf = Sequential()
clf.add(Dense(50, input_dim=n_features, activation='relu'))
clf.add(Dense(50, activation='relu'))
clf.add(Dense(50, activation='relu'))
clf.add(Dense(50, activation='relu'))
clf.add(Dense(50, activation='relu'))
clf.add(Dense(50, activation='relu'))
clf.add(Dense(3, activation='softmax'))

clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
# Fit model with the complete dataset
clf.fit(x, output_y, epochs=1000, batch_size=10)

# Evaluate model with cross-validation
kf = StratifiedKFold(n_splits=5, shuffle = True)

cv_y_test = []
cv_y_pred = []

for train_index, test_index in kf.split(x, y):

    # Separate training and test data
    x_train = x[train_index, :]
    y_train = y[train_index]
    y_train_categorical = to_categorical(y_train)
    
    x_test = x[test_index, :]
    y_test = y[test_index]
    y_test_categorical = to_categorical(y_test)
    
    # Training phase
    clf_cv = Sequential()
    clf_cv.add(Dense(50, input_dim=n_features, activation='relu'))
    clf_cv.add(Dense(50, activation='relu'))
    clf_cv.add(Dense(50, activation='relu'))
    clf_cv.add(Dense(50, activation='relu'))
    clf_cv.add(Dense(50, activation='relu'))
    clf_cv.add(Dense(3, activation='softmax'))
    clf_cv.compile(loss='categorical_crossentropy', optimizer='adam') 
    clf_cv.fit(x_train, y_train_categorical, validation_data= (x_test, y_test_categorical), epochs=1000, batch_size=10)    

    # Test phase    
    y_pred = np.argmax(clf_cv.predict(x_test), axis=-1)  

    cv_y_test.append(y_test)
    cv_y_pred.append(y_pred)


print(classification_report(np.concatenate(cv_y_test), np.concatenate(cv_y_pred)))

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------