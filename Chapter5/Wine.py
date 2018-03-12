import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
    header=None
)
# クラスラベルは三種類で使用したブドウによる分類である。
df_wine.columns = ['Class label', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinty of Ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',   'proline']


def getWineRawData():
    return df_wine


def getWineData(kind='std'):
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    if kind == 'none':
        return X_train, X_test, y_train, y_test
    elif kind == 'norm':
        mms = MinMaxScaler()
        X_train_norm = mms.fit_transform(X_train)
        X_test_norm = mms.fit_transform(X_test)
        return X_train_norm, X_test_norm, y_train, y_test
    elif kind == 'std':
        stdsc = StandardScaler()
        X_train_std = stdsc.fit_transform(X_train)
        X_test_std = stdsc.fit_transform(X_test)
        return X_train_std, X_test_std, y_train, y_test
