from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def kmeans_init_center(X, k):
    # init k center
    return X[np.random.choice(X.shape[0], k, replace=False)]


def kmeans_assign_label(X, centers):
    # assign label after determine center
    D = cdist(X, centers)
    return np.argmin(D, axis=1)


def kmeans_update_centers(X, label, K):
    # calculate center after assign label
    # ------------------------------------
    # init array center K member with value == 0
    centers = np.zeros((K, X.shape[1]))

    # assign ???
    for k in range(K):
        # get all point has label is k
        Xk = X[label == k, :]
        # get center by calculate averger of all point in cluster
        centers[k, :] = np.mean(Xk, axis=0)
    return centers

# return true if new center is match with old center


def has_converged(centers, new_centers):

    return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers])


def kmeans(X, K):
    centers = [kmeans_init_center(X, K)]
    labels = []
    it = 0
    while True:
        # get new label with recenter center
        newLabels = kmeans_assign_label(X, centers[-1])
        # print('new labels: ', newLabels)
        labels.append(newLabels)
        # get new center with new label
        new_centers = kmeans_update_centers(X, labels[-1], K)
        # if last center is match with new centers then break
        if has_converged(centers[-1], new_centers):
            break
        # and new center to centers list to use for next loop
        centers.append(new_centers)
        # print('Centers: ', centers)
        it += 1

    return (centers, labels, it)


def kmeans_display(X, label):
    K = np.amax(label) + 1

    X0 = X[label == 0, :]
    X1 = X[label == 1, :]
    X2 = X[label == 2, :]

    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=2, alpha=0.8)
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=2, alpha=0.8)
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=2, alpha=0.8)

    plt.axis('equal')
    plt.plot()
    plt.show()


# init seed random
np.random.seed(11)

# init center
means = [[2, 2], [0, 3], [3, 6]]
cov = [[1, 0], [0, 1]]  # coordinate system
N = 100  # count init point

# init array point random for every means center
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)

K = 3

original_label = np.asarray([0] * N + [1] * N + [2] * N).T

(centers, labels, it) = kmeans(X, K)
kmeans_display(X, labels[-1])

# cac dieu can chu y o day la:
# - cach random ra point around anchor in a system coordinates
# - cach random ra sub array of array with k member