import numpy as np
import matplotlib.pyplot as plt
from BCW import get_cancer_data
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.learning_curve import validation_curve

pipe_lr = Pipeline(
    [('scl', StandardScaler()), ('clf', LogisticRegression(random_state=0))])

X_train, X_test, y_train, y_test = get_cancer_data()

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr,
                                             X=X_train,
                                             y=y_train,
                                             param_name='clf__C',
                                             param_range=param_range,
                                             cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean, color='blue', marker='o',
         markersize=5, label='Training Accuracy')
plt.fill_between(param_range, train_mean + train_std,
                 train_mean-train_std, alpha=0.15, color='blue')
plt.plot(param_range, test_mean, color='green', marker='s',
         markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(param_range, test_mean + test_std,
                 test_mean-test_std, alpha=0.15, color='green')

plt.grid()
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.ylim([0.8, 1.0])
plt.legend(loc='lower right')
plt.show()
