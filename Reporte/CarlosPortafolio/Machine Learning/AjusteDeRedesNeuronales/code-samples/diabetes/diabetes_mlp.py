#------------------------------------------------------------------------------------------------------------------
#   Multilayer perceptron for the diabetes dataset
#------------------------------------------------------------------------------------------------------------------

from sklearn import datasets
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Import Diabetes dataset
diabetes = datasets.load_diabetes()
x = diabetes.data
y = diabetes.target
features = diabetes.feature_names
n_features = len(features)

# Train model
regr = MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=10000) 
regr.fit(x, y)

y_pred = regr.predict(x)
print('MSE: ', mean_squared_error(y, y_pred))

# 5-fold cross-validation
n_splits=5
kf = KFold(n_splits=n_splits, shuffle = True)

mse = 0
for train_index, test_index in kf.split(x):

    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]

    regr_cv = MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=10000) 
    regr_cv.fit(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]    

    y_pred = regr_cv.predict(x_test)

    # Calculate MSE
    mse_i = mean_squared_error(y_test, y_pred)
    print('mse = ', mse_i)

    mse += mse_i 

mse = mse/n_splits
print('MSE = ', mse)


#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------