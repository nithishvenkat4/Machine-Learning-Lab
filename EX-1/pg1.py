import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("C:\\Users\\admin\\Desktop\\SEM-IV\\ml-lab\\EX-1\\insurance.csv")

#head rows
print(data.columns)
#no of row col
print(len(data.columns))
print(len(data.index))
print(data.shape)

#scatter plot bmi vs charges
sns.scatterplot(data=data,x='bmi',y='charges')
plt.title("BMI vs Charges")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.show()

#scatter plot bmi vs charges color
sns.scatterplot(data=data,x='bmi',y='charges',hue='smoker')
plt.title("BMI vs Charges")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.show()

#LINE PLOT
sns.lineplot(data=data,x='age',y='bmi')
plt.title("Age vs BMI")
plt.ylabel("BMI")
plt.xlabel("Age")
plt.show()

#bar Plot
sns.barplot(data=data,x='children',y='charges')
plt.title("Children VS Charges")
plt.xlabel("Children")
plt.ylabel("Charges")
plt.show()

#corplot
corr=data[['children','age']].corr()
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.title(" Correlation btw Children ,Age")
plt.show()