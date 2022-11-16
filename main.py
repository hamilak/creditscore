import pandas as pd
import numpy as np

# loading csv file
df = pd.read_csv('credit.csv')
# print(df.head(5))

cols = ['credit_history', 'purpose', 'personal_status', 'property', 'other_debtors', 'age', 'installment_plan', 'housing',
     'dependents', 'telephone', 'foreign_worker']
new_df = df.drop(cols, axis='columns')

# print(new_df.head(5))
# print(new_df.info())
print(new_df.isnull)
print(new_df.isna().any().sum())

new_df = new_df.dropna()
print(new_df.isnull)
print(new_df.isna().any().sum())

