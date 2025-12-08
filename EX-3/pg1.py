import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data=pd.read_csv("C:\\Users\\admin\\Desktop\\SEM-IV\\Machine-Learning-Lab\\EX-3\\NFL Play by Play 2009-2016 (v3).csv")

print(data.isnull().sum())
print("Summary")
print(data.info())