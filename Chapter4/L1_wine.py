from Wine import getWineData
import numpy as np
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l', C=0.1)
X_train_std, X_test_std, y_train_std, y_test_std = getWineData()
lr.fit(X_train_std, y_train_std)
print('Training Accuracy: ', lr.score(X_train_std, y_train_std))
print('Test Accuracy: ', lr.score(X_test_std, y_test_std))
print('Intercept:\n', lr.intercept_)
print('Coef:\n', lr.coef_)
