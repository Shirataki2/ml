from Wine import getWineData
from Util import plot_decision_regions
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

X_train_std, X_test_std, y_train, y_test = getWineData()

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
lr = LogisticRegression()
lr.fit(X_train_lda, y_train)
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()
