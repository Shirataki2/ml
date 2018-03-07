from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from Util import plot_decision_regions
from IrisData import getIrisData

X_train, X_test, X_combined, y_train, y_test, y_combined = getIrisData(
    standardized=False)

forest = RandomForestClassifier(
    criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)

forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined,
                      classifier=forest, test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.show()
