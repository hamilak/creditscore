import pandas as pd
import sklearn
import numpy as np

# loading csv file
from sklearn.model_selection import train_test_split

df = pd.read_csv('credit.csv')

# dropping columns
cols = ['credit_history', 'purpose', 'personal_status', 'property', 'other_debtors', 'age', 'installment_plan', 'housing', 'dependents', 'telephone', 'foreign_worker']
new_df = df.drop(cols, axis='columns')
# print(new_df.head(5))
print(new_df.info())
# print(new_df.isnull)
print(new_df.isna().any().sum())
# # drop rows with missing values
new_df = new_df.dropna()
# print(new_df.isnull)
# print(new_df.isna().any().sum())


