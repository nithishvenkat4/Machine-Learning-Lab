import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data=pd.read_csv("C:\\Users\\admin\\Desktop\\SEM-IV\\Machine-Learning-Lab\\EX-3\\NFL Play by Play 2009-2016 (v3).csv")

null_count=data.isnull().sum()
print(null_count)
null_prec=(null_count/len(data)*100)
print(null_prec.sort_values(ascending=False))
print("Summary")
print(data.info())

data_list=data.dropna()
print("Original Shape:",data.shape)
print("Data Shape After Listwise:",data_list.shape)

numeric_cols=data.select_dtypes(include=['int64','float64'])
data_pair=numeric_cols.dropna()
pair_cor=data_pair.corr()
print("Corelation is ",pair_cor)
print("Original Shape:",data.shape)
print("Data Shape After Pairwise:",data_pair.shape)

data_clean=data.copy()

null_col_per=(data_clean.isnull().sum()/len(data))*100
col_to_del=null_col_per[null_col_per>60].index
data_clean=data_clean.drop(columns=col_to_del)
print("Original Shape",data.shape)
print("Shape After Cleaning",data_clean.shape)

numeric_cols=data_clean.select_dtypes(include=['int64','float64']).columns
data_clean[numeric_cols]=data_clean[numeric_cols].fillna(data_clean[numeric_cols].mean())
print("Original Missing Count",data.isnull().sum().sum())
print("Missing Count After Step1 Filling",data_clean.isnull().sum().sum())

data_clean[numeric_cols]=data_clean[numeric_cols].fillna(data_clean[numeric_cols].median())
print("Original Missing Count",data.isnull().sum().sum())
print("Missing Count After Step2 Filling",data_clean.isnull().sum().sum())

catg_col=data_clean.select_dtypes(include='object').columns

for col in catg_col:
    mod_val=data_clean[col].mode()
    if(len(mod_val)>0):
        data_clean[col]=data_clean[col].fillna(mod_val[0])

print("Original Missing Count",data.isnull().sum().sum())
print("Missing Count After Step 3 Filling",data_clean.isnull().sum().sum())
