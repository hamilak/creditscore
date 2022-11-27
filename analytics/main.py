import pandas as pd
import numpy as np

# loading csv file
df = pd.read_csv('credit.csv')

# male to female ratio
females = df['gender'].value_counts()['female']
males = df['gender'].value_counts()['male']

ratio = (males/females)
print(f'The ratio of male workers to female workers is {ratio}')

# proportion of the population who own a real estate property are also citizens
total_owners = len(df['property'])
real_estate_owners = len(df.query('property == "real estate" & foreign_worker == "no"'))
proportion = (real_estate_owners / total_owners)
print(f'The proportion of the population who own a real estate property are also citizens is {proportion}')

# the percentage of married people that are dependents have borrowed for business
# purpose, but have also defaulted on their loan

defaulting_married_people = len(df.query('personal_status== "married" & purpose == "business" & default == 1 '))

total_married_people = len(df['personal_status'])
percentage = (defaulting_married_people/total_married_people) * 100

print(f'The percentage of married people that are dependents have borrowed for business'
      f'purpose, but have also defaulted on their loan is {percentage}%')

#  total amount of Male customers with more that -100 checking balance have borrowed in the past

defaulting_male_customers = len(df.query('gender == "male" & checking_balance < -100 '))

print(f'There are {defaulting_male_customers} number of customers with more than'
      f'-100 checking balance borrowed in the past')

# Comparison of duration it takes Female customers between the age of 30 – 50 to repay their loan to male customers

female_customers = df.query('gender == "female" & 30 > age < 50')['months_loan_duration'].sum()
male_customers = df.query('gender == "male" & 30 > age < 50')['months_loan_duration'].sum()
print(female_customers)
