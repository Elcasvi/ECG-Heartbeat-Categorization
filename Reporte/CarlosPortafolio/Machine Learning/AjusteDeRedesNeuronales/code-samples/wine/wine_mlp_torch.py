#------------------------------------------------------------------------------------------------------------------
#   Multilayer perceptron for the Wine dataset (PyTorch)
#------------------------------------------------------------------------------------------------------------------

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

# Load data
wine = datasets.load_wine()
x = wine.data
y = wine.target

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Define model
class WineMLP(nn.Module):
    
    def __init__(self):
        super(WineMLP, self).__init__()
        self.layer1 = nn.Linear(13, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 20)
        self.layer4 = nn.Linear(20, 20)
        self.layer5 = nn.Linear(20, 20)
        self.output_layer = nn.Linear(20, 3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = self.output_layer(x)
        return x

# Define loss function
loss_function = nn.CrossEntropyLoss()
    
# Create model
clf = WineMLP()

# Train model with the complete dataset
optimizer = optim.Adam(clf.parameters(), lr=0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    
    # Fordward phase
    outputs = clf(x)
    loss = loss_function(outputs, y)
    
    # Backward phase and optimization of parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print results
    if (epoch+1) % 10 == 0:
        print("Epoch: {}, Loss: {}".format(epoch+1, loss.item()))

# Evaluate model with cross-validation
kf = StratifiedKFold(n_splits=5, shuffle = True)

cv_y_test = []
cv_y_pred = []

for train_index, test_index in kf.split(x, y):

    # Separate training and test data
    x_train = x[train_index, :]
    y_train = y[train_index]
    
    x_test = x[test_index, :]
    y_test = y[test_index]
    
    # Training phase
    clf_cv = WineMLP()
    optimizer = optim.Adam(clf_cv.parameters(), lr=0.001)    
    
    for epoch in range(num_epochs):         
        outputs = clf_cv(x_train)
        loss = loss_function(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        if (epoch+1) % 10 == 0:
            print("Epoch: {}, Loss: {}".format(epoch+1, loss.item()))
            
    # Test phase
    clf_cv.eval()
    with torch.no_grad():
        outputs = clf_cv(x_test)
        _, y_pred = torch.max(outputs.data, 1)
    
    cv_y_test.append(y_test)
    cv_y_pred.append(y_pred)


print(classification_report(np.concatenate(cv_y_test), np.concatenate(cv_y_pred)))

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------