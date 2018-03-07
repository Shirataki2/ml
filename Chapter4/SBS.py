from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class SBS(object):
    # 逐次後退選択(Sequential Backward Selection)

    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring              # 特徴量を評価する指標
        self.estimator = clone(estimator)   # 推定器
        self.k_features = k_features        # 選択する特徴量の個数
        self.test_size = test_size          # テストデータの割合
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1]