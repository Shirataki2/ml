import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'],
    ['blue', 'L', 13.5, 'class2'],
    ['blue', 'XL', 17.5, 'class1'],
    ['blue', 'M', 7.5, 'class2'],
    ['purple', 'XL', 17.1, 'class1'],
    ['purple', 'L', 14.1, 'class1'],
    ['purple', 'M', 8.7, 'class2'],
    ['gray', 'M', 9.4, 'class2']
])
df.columns = ['color', 'size', 'price', 'class label']
print(df)
print('-'*30)

size_mapping = {'M': 1, 'L': 2, 'XL': 3}
df['size'] = df['size'].map(size_mapping)
class_mapping = {label: idx for idx,
                 label in enumerate(np.unique(df['class label']))}
df['class label'] = df['class label'].map(class_mapping)
inv_class_mapping = {v: k for k, v in class_mapping.items()}
print(df)
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print('-'*30)
print(X)

ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()
print('-'*30)
print(X)
