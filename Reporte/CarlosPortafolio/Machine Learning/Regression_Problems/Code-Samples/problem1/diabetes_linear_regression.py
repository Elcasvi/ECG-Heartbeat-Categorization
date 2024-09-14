#------------------------------------------------------------------------------------------------------------------
#   Linear model for the diabetes dataset
#------------------------------------------------------------------------------------------------------------------

from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import Diabetes dataset
diabetes = datasets.load_diabetes()
x = diabetes.data
y = diabetes.target
features = diabetes.feature_names
n_features = len(features)

# Train linear regression model using the complete data set
regr = linear_model.LinearRegression()
regr.fit(x, y)
print("Coeficientes del modelo: ", regr.coef_)

y_pred = regr.predict(x)
print('MSE: ', mean_squared_error(y, y_pred))
print('MAE: ', mean_absolute_error(y, y_pred))
print("R^2: ", r2_score(y, y_pred))

# k-fold cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True)

mse = 0
mae = 0
r2 = 0
for train_index, test_index in kf.split(x):
    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]

    regr_cv = linear_model.LinearRegression()
    regr_cv.fit(x_train, y_train)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]

    y_pred = regr_cv.predict(x_test)

    # Calculate MSE and R^2    
    mse_i = mean_squared_error(y_test, y_pred)
    print('mse = ', mse_i)

    msa_i = mean_absolute_error(y_test, y_pred)
    print('msa = ', mse_i)

    r2_i = r2_score(y_test, y_pred)
    print('r^2= ', r2_i)

    mse += mse_i
    r2 += r2_i

mse = mse / n_folds
print('MSE = ', mse)

mse = mae / n_folds
print('MAE = ', mse)

r2 = r2 / n_folds
print('R^2 = ', r2)

# k-fold cross-validation using cross_val_predict
n_folds = 5
regr = linear_model.LinearRegression()
y_pred = cross_val_predict(regr, x, y, cv=n_folds)

print('mse = ', mean_squared_error(y, y_pred))

print('msa = ', mean_absolute_error(y, y_pred))

print('r^2= ', r2_score(y, y_pred))

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------
