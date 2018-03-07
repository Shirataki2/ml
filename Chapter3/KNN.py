from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from Util import plot_decision_regions
from IrisData import getIrisData

X_train_std, X_test_std, X_combined_std, y_train, y_test, y_combined = getIrisData()

nb = 5
p = 2

knn = KNeighborsClassifier(n_neighbors=nb, p=p, metric='minkowski')

knn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=knn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.title("KNN ($neighbors=%d,p=%d $)" % (nb, p))
plt.legend(loc='upper left')

plt.show()
