import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron(object):

    def __init__(self, eta=0.10, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    """
    X   arraylike   2d  row:sample  col:feature
    y   arraylike   1d  target

    【MEMO】
    慣例として初期化時に設定されるselfの変数(メンバ変数)以外は_を後ろにつける
    """

    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1] + 1)
        self.errors_ = []

        for _ in range(self.n_iter):
            error = 0  # 誤分類数
            for xi, target in zip(X, y):
                update = self.eta*(target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                error += int(update != 0.0)
            self.errors_.append(error)
        return self

    def net(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    """
    input   X   arraylike   1d  sample
    output  1 or -1
    """

    def predict(self, X):
        return np.where(self.net(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier=Perceptron, area=plt, resolution=0.02):
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

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    # setosaのプロット
    axs[0, 0].scatter(X[:50, 0], X[:50, 1], c='red',
                      marker='o', label='setosa')
    axs[0, 0].scatter(X[50:100, 0], X[50:100, 1], c='blue',
                      marker='x', label='versicolor')
    axs[0, 0].set_xlabel('sepal length [cm]')  # 萼片
    axs[0, 0].set_ylabel('petal length [cm]')  # 花弁
    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    axs[1, 0].plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Number of Misclassifications')
    plot_decision_regions(X, y, ppn, axs[0, 1])
    axs[0, 1].set_xlabel('sepal length [cm]')  # 萼片
    axs[0, 1].set_ylabel('petal length [cm]')  # 花弁
    plt.draw()
    plt.show()
