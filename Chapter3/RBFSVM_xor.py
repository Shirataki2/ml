import matplotlib.pyplot as plt
import numpy as np
from Util import plot_decision_regions
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

gamma = 1

svm = SVC(kernel='rbf', C=10.0, random_state=0, gamma=gamma)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)

plt.legend(loc='upper left')
plt.title("RBF Kernel SVM ($\gamma $ = %f)" % gamma)
plt.show()
