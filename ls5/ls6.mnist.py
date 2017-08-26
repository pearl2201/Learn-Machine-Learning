from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import time
from mnist import MNIST

mndata = MNIST('E:\Anh Ngoc 2\Machine Learning\ls5\mnist')
mndata.load_testing()
mndata.load_training()
X_test = mndata.test_images
y_test = np.asarray(mndata.test_labels)
X_train = mndata.train_images
y_train = np.asarray(mndata.train_labels)

startTime = time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors=1, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end_time = time.time()

print("Accuracy of 1NN for MNIST: %.2f %%" %
      (100 * accuracy_score(y_test, y_pred)))
print("Running time %.2f (s) " % (end_time - startTime))
