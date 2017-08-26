from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

(iris_X, iris_y) = datasets.load_iris(return_X_y=True)

print('Number of classes: %d' % len(np.unique(iris_y)))
print('Number of data points: %d' % len(iris_y))

X0 = iris_X[iris_y == 0, :]
print('X0: ', X0)
print('\nSample from class 0:\n', X0[:5, :])

X1 = iris_X[iris_y == 1, :]
print('X1: ', X1)
print('\nSample from class 1: \n', X1[:5, :])

X2 = iris_X[iris_y == 2, :]
print('X2: ', X2)
print('\nSample from class 2: \n', X2[:5, :])

X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size=50)

print("Training size: %d" % len(y_train))
print("Test size: %d" % len(y_test))

clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy of 1NN: %.2f %%" % (100 * accuracy_score(y_test, y_pred)))

clf = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights='distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy of 10NN: %2f %%" % (100 * accuracy_score(y_test, y_pred)))
