import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.random import seed


class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initalized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        self._initalize_weights(X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initalized:
            self._initalize_weights(X.shape[1])
        if y.rabel().shape[0] > 0:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initalize_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initalized = True

    def _update_weights(self, xi, target):
        output = self.net(xi)
        error = (target-output)
        self.w_[1:] += self.eta*xi.dot(error)
        self.w_[0] += self.eta*error
        cost = 0.5*error**2
        return cost

    # 活性化関数(この場合は恒常関数)
    def activation(self, X):
        return self.net(X)

    def net(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        input   X   arraylike   1d  sample
        output  1 or -1
        """
        return np.where(self.net(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier=AdalineSGD, area=plt, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()+1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    area.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    area.set_xlim(xx1.min(), xx1.max())
    area.set_ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        area.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                     c=cmap(idx), marker=markers[idx], label=cl)


if __name__ == '__main__':
    df = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    # Iris-setosa       -1
    # Iris-virginica    +1
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    X_std = np.copy(X)
    X_std[:, 0] = (X_std[:, 0] - X_std[:, 0].mean()) / X_std[:, 0].std()
    X_std[:, 1] = (X_std[:, 1] - X_std[:, 1].mean()) / X_std[:, 1].std()

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))

    ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1).fit(X_std, y)
    plot_decision_regions(X_std, y, ada, ax1)
    ax1.set_xlabel('sepal length[standardized]')
    ax1.set_ylabel('petal length[standardized]')
    ax1.set_title('Adaline - Stochastic Gradient Descent')
    ax1.legend(loc='upper left')

    ax2.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Sum-Squared-Error')
    plt.draw()
    plt.show()
