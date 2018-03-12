from KernelPCA import rbf_kernel_pca
from matplotlib.ticker import FormatStrFormatter
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

X, y = make_circles(n_samples=1000, noise=0.1, factor=0.2, random_state=123)
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
ax[1].scatter(X_spca[y == 0, 0], np.zeros((500, 1)) + 0.02,
              color='r', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((500, 1)) - 0.02,
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
ax[1].scatter(X_kpca[y == 0, 0], np.zeros((500, 1)) + 0.02,
              color='r', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros((500, 1)) - 0.02,
              color='b', marker='o', alpha=0.5)
ax[0].set_xlabel('PC 1')
ax[1].set_xlabel('PC 1')
ax[0].set_ylabel('PC 2')
ax[1].set_yticks([])
ax[1].set_ylim([-1, 1])
plt.show()
