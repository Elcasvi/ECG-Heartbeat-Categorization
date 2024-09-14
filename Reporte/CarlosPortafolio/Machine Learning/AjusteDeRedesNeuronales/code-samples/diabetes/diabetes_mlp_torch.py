#------------------------------------------------------------------------------------------------------------------
#   Multilayer perceptron for the diabetes dataset (PyTorch)
#------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Import Diabetes dataset
diabetes = datasets.load_diabetes()
x = diabetes.data
y = diabetes.target
features = diabetes.feature_names
n_features = len(features)

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Define model
class DiabetesMLP(nn.Module):
    
    def __init__(self):
        super(DiabetesMLP, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 20)
        self.layer4 = nn.Linear(20, 20)
        self.layer5 = nn.Linear(20, 20)
        self.output_layer = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = self.output_layer(x)
        return x

# Define loss function
loss_function = nn.MSELoss()

# Create model
regr = DiabetesMLP()

# Train model with the complete dataset
optimizer = optim.Adam(regr.parameters(), lr=0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    
    # Fordward phase
    outputs = regr(x).squeeze()
    loss = loss_function(outputs, y)
    
    # Backward phase and optimization of parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print results
    if (epoch+1) % 10 == 0:
        print("Epoch: {}, Loss: {}".format(epoch+1, loss.item()))

# Set model to evaluation mode and predict a new observation
regr.eval()

with torch.no_grad():
    y_pred = regr(x).squeeze()   
    
print('MSE: ', loss_function(y_pred, y))

# 5-fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle = True)

mse = 0
for train_index, test_index in kf.split(x):
    
    # Separate training and test data
    x_train = x[train_index, :]
    y_train = y[train_index]

    x_test = x[test_index, :]
    y_test = y[test_index]  

    # Training phase
    regr_cv = DiabetesMLP()
    optimizer = optim.Adam(regr_cv.parameters(), lr=0.001)    
    
    for epoch in range(num_epochs):         
        outputs = regr_cv(x_train).squeeze()
        loss = loss_function(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        if (epoch+1) % 10 == 0:
            print("Epoch: {}, Loss: {}".format(epoch+1, loss.item()))

    # Test phase
    regr_cv.eval()
    with torch.no_grad():
        y_pred = regr_cv(x_test).squeeze()
 
    mse_i = mean_squared_error(y_test, y_pred)
    mse += mse_i 

    print('mse = ', mse_i)    

mse = mse/n_splits
print('MSE = ', mse)

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------