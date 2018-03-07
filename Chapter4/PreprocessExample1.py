import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer
csv_data = """A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,7.0,
9.0,,11.0,
13.0,14.0,,16.0"""
df = pd.read_csv(StringIO(csv_data))
print(df)
print(df.isnull().sum())
# 欠測値を含む行を消去
print(df.dropna())
# 欠測値を含む列を消去
print(df.dropna(axis=1))
print('-'*20)
print("欠測値の補完\n")
imr = Imputer(strategy='mean', axis=0)
imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)
