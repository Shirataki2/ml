import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


class AdalineGD(object):
    def __init__(self, eta=0.10, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        input   X   arraylike   2d
                y   arraylike   1d
        output  this object
        """
        self.w_ = np.zeros(X.shape[1]+1)
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net(X)
            errors = (y-output)
            self.w_[1:] += self.eta*X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

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


def plot_decision_regions(X, y, classifier=AdalineGD, area=plt, resolution=0.02):
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

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0, 0].plot(range(1, len(ada1.cost_)+1),
                  np.log10(ada1.cost_), marker='o')
    ax[0, 0].set_xlabel('Epochs')
    ax[0, 0].set_ylabel('log(Sum-Squared-Error)')
    ax[0, 0].set_title('Adaline - Learning Rate = 0.01')

    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[0, 1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
    ax[0, 1].set_xlabel('Epochs')
    ax[0, 1].set_ylabel('Sum-Squared-Error')
    ax[0, 1].set_title('Adaline - Learning Rate = 0.0001')

    # 正規化
    X_std = np.copy(X)
    X_std[:, 0] = (X_std[:, 0] - X_std[:, 0].mean()) / X_std[:, 0].std()
    X_std[:, 1] = (X_std[:, 1] - X_std[:, 1].mean()) / X_std[:, 1].std()
    ada3 = AdalineGD(n_iter=15, eta=0.01).fit(X_std, y)
    plot_decision_regions(X_std, y, ada3, ax[1, 0])
    ax[1, 0].set_xlabel('sepal length[standardized]')
    ax[1, 0].set_ylabel('petal length[standardized]')
    ax[1, 0].set_title('Adaline - Gradient Descent')
    ax[1, 0].legend(loc='upper left')
    ax[1, 1].plot(range(1, len(ada3.cost_)+1), ada3.cost_, marker='o')
    ax[1, 1].set_xlabel('Epochs')
    ax[1, 1].set_ylabel('Sum-Squared-Error')
    ax[1, 1].set_title('Adaline[standardized] - Learning Rate = 0.01')
    plt.draw()
    plt.show()
