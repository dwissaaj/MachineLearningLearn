import math as mt
import pandas as pd
import quandl
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression

#IHSG 2020 - 2021 PREDICTTION
df = pd.read_csv('IHSG-2020-2021.csv')

df['HL_PCT'] = (df['High'] - df['Low']) / df['Low'] * 100

df['PCT_Change'] = (df['Close'] - df['Open']) / df['Close'] * 100

df = df[['Adj Close', 'HL_PCT', 'PCT_Change', 'Volume']]
forecast_col = 'Adj Close'
df.fillna(-99999, inplace=True)
forecast_out = int(mt.ceil(0.01 * len(df)))
df['Label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['Label'], 1))
y = np.array(df['Label'])
X = preprocessing.scale(X)
y = np.array(df['Label'])


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
fit1 = clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)