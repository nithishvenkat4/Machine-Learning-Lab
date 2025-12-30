#indian diabetes

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

data=pd.read_csv("C:\\Users\\admin\\Desktop\\SEM-IV\\Machine-Learning-Lab\\EX-8\\diabetes.csv")

print(data.shape)
print(data.columns)
print(data.head())
print(data.isnull().sum())

print(data["Outcome"].value_counts())
print((data[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin","BMI", "DiabetesPedigreeFunction", "Age"]]==0).sum())

cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[cols_to_fix]=data[cols_to_fix].replace(0,data[cols_to_fix].median())

print((data[cols_to_fix]==0).sum())

X=data.drop(columns=["Outcome"])
y=data["Outcome"]
print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=35,stratify=y)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion="gini",max_depth=3,min_samples_leaf=15,random_state=40)

model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print("Accuracy Score :",accuracy_score(y_test,y_pred))
print("Confusion Matrix :\n",confusion_matrix(y_test,y_pred))
print("Classification Report :",classification_report(y_test,y_pred))

train_score = model.score(X_train,y_train)
test_score = model.score(X_test,y_test)

print("Model Score of Tain Set :",train_score)
print("Model Score of Test Set :",test_score)

from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Non-Diabetic", "Diabetic"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.show()
