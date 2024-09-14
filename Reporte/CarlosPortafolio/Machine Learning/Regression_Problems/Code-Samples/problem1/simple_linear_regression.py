#------------------------------------------------------------------------------------------------------------------
#   Simple linear regression example
#------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.random.uniform(0, 10, 100)
y = 5  + 2*x + np.random.normal(0, 1, 100)

# Plot the data
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Find model coefficients
n = len(x)
a = ( (x*y).sum()-(1./n)*x.sum()*y.sum() )/( (x*x).sum() - (1./n)*x.sum()**2)
b = (1./n)*y.sum() - (a/n)*x.sum()
print ("Model: ", b, "+", a, "x")

# Evaluate residuals
y_pred = a*x + b
r = y - y_pred

# Plot model
plt.scatter(x, y)
plt.axline((0, b), slope = a, color = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Model')
plt.show()

# Plot residuals
plt.scatter(y, r)
plt.axline((0, 0), slope = 0, color = 'red')
plt.xlabel('y')
plt.ylabel('Error')
plt.title('Residuals')
plt.show()

#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------
