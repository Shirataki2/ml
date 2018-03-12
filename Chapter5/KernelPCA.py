from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt

"""
scipy.spatial.distance.pdist(X, 'sqeuclidean')
X (n x m) の二乗ユークリッド距離行列を返す
正方距離行列に変換するにはsquareformを使う
"""


def rbf_kernel_pca(X, gamma, n_components):
    sq_dists = pdist(X, 'sqeuclidean')

    mat_sq_dists = squareform(sq_dists)

    K = exp(-gamma*mat_sq_dists)

    N = K.shape[0]
    one_n = np.ones((N, N))/N
    K = K-one_n.dot(K)-K.dot(one_n)+one_n.dot(K).dot(one_n)
    _, eigvecs = eigh(K)

    X_pc = np.column_stack((eigvecs[:, -i]
                            for i in range(1, n_components + 1)))
    return X_pc


if __name__ == '__main__':
    X, y = make_moons(n_samples=100, random_state=123)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='r', marker='^', alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='b', marker='o', alpha=0.5)
    plt.show()
    sk_pca = PCA(n_components=2)
    X_spca = sk_pca.fit_transform(X)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
    ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],
                  color='r', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],
                  color='b', marker='o', alpha=0.5)
    ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02,
                  color='r', marker='^', alpha=0.5)
    ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02,
                  color='b', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC 1')
    ax[1].set_xlabel('PC 1')
    ax[0].set_ylabel('PC 2')
    ax[1].set_yticks([])
    ax[1].set_ylim([-1, 1])
    plt.show()
    X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
    ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1],
                  color='r', marker='^', alpha=0.5)
    ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1],
                  color='b', marker='o', alpha=0.5)
    ax[1].scatter(X_kpca[y == 0, 0], np.zeros((50, 1)) + 0.02,
                  color='r', marker='^', alpha=0.5)
    ax[1].scatter(X_kpca[y == 1, 0], np.zeros((50, 1)) - 0.02,
                  color='b', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC 1')
    ax[1].set_xlabel('PC 1')
    ax[0].set_ylabel('PC 2')
    ax[1].set_yticks([])
    ax[1].set_ylim([-1, 1])
    plt.show()
