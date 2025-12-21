#slr model building 

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data=pd.read_csv("A:\\Users\\Hp\\Documents\\PROGRAMMIING\\COLLEGE\\SEM-IV\\Machine-Learning-Lab\\EX-5\\SAT_DS.csv")

print(data.head())
print(data.tail())
print(data.shape)
print(data.columns)

print(data.dtypes)
print(data.isnull().sum())

X= data[["SAT"]]
y= data["GPA"]

model=LinearRegression()
model.fit(X,y)

print("Slope :",model.coef_)
print("Intercept :",model.intercept_)

y_pred=model.predict(X)
print(y_pred)

