from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
# height

X = np.array(
    [[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T

# weight (kg)
y = np.array([[49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# building Xbar
one = np.ones((X.shape[0], 1))
# print('ones = ', one)
Xbar = np.concatenate((one, X), axis=1)


# print('Xbar = ', Xbar)

# scikitlearn
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, y)

# result scikitlearning
print('Solution found by scikit-learn: ', regr.coef_)

# calculating weight of the fitting line
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)

# preparing the fitting line

w_0 = w[0][0]
w_1 = w[1][0]

x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1 * x0


# Drawing the fitting line
plt.plot(X.T, y.T, 'ro')     # data
plt.plot(x0, y0)               # the fitting line
plt.axis([140, 190, 45, 75])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()
