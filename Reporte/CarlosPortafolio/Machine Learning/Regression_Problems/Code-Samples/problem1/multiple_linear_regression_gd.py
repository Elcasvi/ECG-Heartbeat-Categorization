#------------------------------------------------------------------------------------------------------------------
#   Multiple linear regression optimized with gradient descent
#------------------------------------------------------------------------------------------------------------------

import numpy as np
import numpy.linalg as ln
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Gradient of error function (it evaluates the gradient of the mean squared error function for the specified model and data set).
def grad(X, y, beta):
    n = len(y)
    y_pred = X @ beta
    res = y - y_pred    
    tmp = res*X.transpose()
    return -(2/n)*tmp.sum(axis = 1)

# Predict function (it evaluates an array of observations using the specified linear model).
def predict(X, beta):
    return X @ beta

# Fit model function (it fits a linear model using the specified data set).
def fit_model(X, y, alpha = 0.005, maxit = 10000):
    
    # Number of predictors
    npredictors = X.shape[1]    

    # Initialize beta
    beta = 2*np.random.rand(npredictors) - 1.0
    
    # Optimization algorithm
    it = 0    
    while (ln.norm(grad(X, y, beta)) > 1e-4) and (it < maxit):    
        beta = beta - alpha*grad(X, y, beta)
        it = it + 1
        #print(beta)

    return beta    

# Generate data
x = np.random.uniform(0, 10, (100, 4))
X = np.column_stack((np.ones(x.shape[0]), x))

beta_real = [5, -3, 4, 15, 10]

y = X @ beta_real + np.random.normal(0, 1, 100)

# Build linear model
beta = fit_model(X, y)
print ("Model coefficients: ", beta)

# Evaluate residuals
y_pred = predict(X, beta)
r = y - y_pred

# Plot residuals
plt.scatter(y, r)
plt.axline((0, 0), slope = 0, color = 'red')
plt.xlabel('y')
plt.ylabel('Error')
plt.title('Residuals')
plt.show()

# Calculate MSE, MAE and R^2 with the training set
print('MSE: ', mean_squared_error(y, y_pred))
print("MAE: ", mean_absolute_error(y, y_pred))
print("R^2: ", r2_score(y, y_pred))

# Evaluate model with cross validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle = True)

mse_cv = []
mae_cv = []
r2_cv = []
for train_index, test_index in kf.split(x):
    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    beta_cv = fit_model(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]
    y_pred = predict(x_test, beta_cv)

    # Calculate MSE, MAE and R^2
    mse_i = mean_squared_error(y_test, y_pred)
    print('mse = ', mse_i)  
    mse_cv.append(mse_i)    

    mae_i = mean_absolute_error(y_test, y_pred)
    print('mae = ', mae_i)    
    mae_cv.append(mae_i)
    
    r2_i = r2_score(y_test, y_pred)
    print('r^2= ', r2_i)    
    r2_cv.append(r2_i)   
    
print('MSE:', np.average(mse_cv), '  MAE:', np.average(mae_cv),'  R^2:', np.average(r2_cv))

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------
