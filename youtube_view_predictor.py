import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import sys

dataset = pd.read_csv('yt.csv')
print(dataset.head())

X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values
Xl = dataset.iloc[:, 0:3].values
yl = dataset.iloc[:, 3].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

import numpy as np
dataset=np.array(dataset)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

regressor.score(X_test,y_test)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

from sklearn.linear_model import LinearRegression
Xl_train, Xl_test, yl_train, yl_test = train_test_split(Xl, yl, test_size=0.2)
model = LinearRegression()
model.fit(Xl_train, yl_train)

yl_pred = model.predict(Xl_test)
print(yl_pred)

model.score(Xl_test,yl_test)

output = pd.DataFrame({'Actual': yl_test.flatten(), 'Predicted': yl_pred.flatten()})
output = output.astype({'Actual':int,'Predicted':int})
print(output)

av=(y_pred+yl_pred)/2

avg=(y_pred+yl_pred)/2
av=pd.DataFrame({'Actual': y_test, 'Predicted': avg})
av=av.astype({'Actual':int,'Predicted':int})
print(av)
graph = av.tail(20)
graph.plot(kind='bar')
plt.show()

