from Wine import getWineData, getWineRawData
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
    header=None
)
# クラスラベルは三種類で使用したブドウによる分類である。
df_wine.columns = ['Class label', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinty of Ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',   'proline']

lr = LogisticRegression(penalty='l', C=0.1)
X_train_std, X_test_std, y_train_std, y_test_std = getWineData()

fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']

weights, params = [], []

for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10.0**c, random_state=0)
    lr.fit(X_train_std, y_train_std)
    weights.append(lr.coef_[1])
    params.append(10.0**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column+1], color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.xlabel('C')
plt.ylabel('weight coefficient')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(
    1.38, 1.03), ncol=1, fancybox=True)
plt.savefig('L1Preprocessing.png')
