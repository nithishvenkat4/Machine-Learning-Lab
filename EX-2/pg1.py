import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("C:\\Users\\admin\\Desktop\\SEM-IV\\ml-lab\\EX-2\\loan_data.csv")

cor=data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term', 'Credit_History']].corr()
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()

sns.scatterplot(data=data,x='ApplicantIncome',y='CoapplicantIncome')
plt.title('Applicant Income Vs Co-Aplicant Income')
plt.show()

sns.scatterplot(data=data,x='ApplicantIncome',y='CoapplicantIncome',size='LoanAmount',hue='Credit_History')
plt.title('Applicant Income Vs Co-Aplicant Income')
plt.show()

#box plot for comkparing 2 incomes
sns.boxplot(data=data,x='ApplicantIncome',y='CoapplicantIncome')
plt.title('Applicant Income Vs Co-Aplicant Income')
plt.show()