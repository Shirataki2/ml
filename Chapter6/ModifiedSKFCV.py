from Pipeline import cancer_pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
import numpy as np

pipe_lr, X_train, X_test, y_train, y_test = cancer_pipeline()

kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)
scores = cross_val_score(estimator=pipe_lr, X=X_train,
                         y=y_train, cv=10)


print('CV Accuracy Scores: %s' % scores)
print('CV Accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
