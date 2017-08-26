from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import time
import mnist


X_test = mnist.test_images()

y_test = mnist.test_labels()
X_train = mnist.train_images()
X_train = X_train.reshape(
    (X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
print("X_train: ", X_train)
y_train = mnist.train_labels()
print("y_train: ", y_train)
startTime = time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end_time = time.time()

print("Accuracy of 1NN for MNIST: %.2f %%" %
      (100 * accuracy_score(y_test, y_pred)))
print("Running time %.2f (s) " % (end_time - startTime))
