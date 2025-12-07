import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("A:\\Users\\Hp\\Documents\\PROGRAMMIING\\COLLEGE\\SEM-IV\\Machine-Learning-Lab\\EX-2\\loan_data.csv")

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
sns.boxplot(data=data,x='Education',y='CoapplicantIncome')
plt.title('Applicant Income Vs Co-Aplicant Income')
plt.show()

#histplot
sns.histplot(data=data[['ApplicantIncome','CoapplicantIncome','LoanAmount']],kde=True)
plt.title("Incomes and Loan Amount")
plt.show()