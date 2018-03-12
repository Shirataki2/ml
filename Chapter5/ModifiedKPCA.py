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
    eigvals, eigvecs = eigh(K)

    '''
     K $al$ = $lm$ $al$
    '''
    alphas = np.column_stack((eigvecs[:, -i]
                              for i in range(1, n_components + 1)))
    lambdas = [eigvals[-i] for i in range(1, n_components + 1)]
    return alphas, lambdas


def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row) ** 2) for row in X])
    k = np.exp(- gamma * pair_dist)
    return k.dot(alphas/lambdas)


if __name__ == '__main__':
    # 新しいデータが射影できているのかを確認する
    X, y = make_moons(n_samples=100, random_state=123)
    alphas, lambdas = rbf_kernel_pca(X, gamma=15, n_components=1)
    x_new = X[25]
    print("New Sample:\n", x_new)
    x_proj = alphas[25]
    print("Expected Eigenvector :", x_proj)
    x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
    print("Result :", x_reproj)
    plt.scatter(alphas[y == 0, 0], np.zeros(
        (50)), c='r', marker='^', alpha=0.5)
    plt.scatter(alphas[y == 1, 0], np.zeros(
        (50)), c='b', marker='o', alpha=0.5)
    plt.scatter(x_proj, 0, c='black',
                label='Original Projection of Point X[25]', marker='^', s=100)
    plt.scatter(x_proj, 0, c='green',
                label='Remapped Point X[25]', marker='x', s=500)
    plt.legend(scatterpoints=1)
    plt.show()
