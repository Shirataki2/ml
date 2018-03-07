from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def getIrisData(standardized=True, testSize=0.3, randomState=0):

    iris = load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testSize, random_state=randomState)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    if standardized:
        return X_train_std, X_test_std, X_combined_std, y_train, y_test, y_combined
    else:
        return X_train, X_test, X_combined, y_train, y_test, y_combined
