#MLR co2  Car Dataset
#dependent CO2
#independent Vol,weight,etc

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


data=pd.read_csv("C:\\Users\\admin\\Desktop\\SEM-IV\\Machine-Learning-Lab\\EX-7\\DATA.csv")
print(data.shape)

print(data.describe())
print(data.head())

print(data.isnull().sum())
data=data.drop(columns=["Unnamed: 5"])
print(data.shape)

plt.scatter(data["Volume"],data["CO2"])
plt.show()
plt.scatter(data["Weight"],data["CO2"])
plt.show()

X = data[["Volume", "Weight"]]
X = sm.add_constant(X)

vif = pd.DataFrame()
vif["Feature"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i)
              for i in range(X.shape[1])]

print(vif)

X = data[["Volume", "Weight"]]
y=data["CO2"]

model=LinearRegression()
model.fit(X,y)

print("Slope :",model.coef_)
print("Intercept:",model.intercept_)

y_pred=model.predict(X)

confusion_matrix=[0,0]
for i in range(0,len(y)):
    if(y[i]==y_pred[i]): confusion_matrix[0]=confusion_matrix[0]+1
    else : confusion_matrix[1]=confusion_matrix[1]+1
print(confusion_matrix)
r2=r2_score(y,y_pred)
mae=mean_absolute_error(y,y_pred)
mse=mean_squared_error(y,y_pred)
rmse=np.sqrt(mse)

print("R Square :",r2)
print("Mean Absolute Error :",mae)
print("Mean Squared Error :",mse)
print("Root Mean Squared Error :",rmse)

X_new=[[2345,1234]]
y_pred_new=model.predict(X_new)
print("New Predicted Value :",y_pred_new)

plt.scatter(y, y_pred)
plt.xlabel("Actual CO2")
plt.ylabel("Predicted CO2")
plt.title("Actual vs Predicted CO2")
plt.show()
