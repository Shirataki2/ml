from sklearn.linear_model import LinearRegression, RANSACRegressor
from LinearRegression import LinearRegressionGD, get_data, lin_regplot
import numpy as np
import matplotlib.pyplot as plt

df = get_data()
X = df[['RM']].values
y = df['MEDV'].values

ransac = RANSACRegressor(LinearRegression(),
                         max_trials=150,
                         min_samples=50,
                         residual_metric=lambda x:
                         np.sum(np.abs(x), axis=1),
                         residual_threshold=5.0,
                         random_state=0)
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='b', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='r', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Average number of Rooms[RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()
