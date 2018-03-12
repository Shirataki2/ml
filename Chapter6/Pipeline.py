from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from BCW import get_cancer_data


def cancer_pipeline():
    X_train, X_test, y_train, y_test = get_cancer_data()

    pipe_lr = Pipeline([('scl', StandardScaler()),
                        ('pca', PCA(n_components=2)),
                        ('clf', LogisticRegression(random_state=1))])
    return pipe_lr, X_train, X_test, y_train, y_test
