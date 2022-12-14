Project: Creditworthiness

You are given a dataset containing demographic as well as personal information of individuals existing on a credit company platform. We want to answer some questions about the data, model the data, and understand the impact our prediction has on business decision making. 

For this task, you are to prepare the solution in Google Colab file (.ipynb) and a python file (.py). 

NB: Make sure you structure your code properly, use the PEP standard in your code, use the proper variable naming, syntax, and make your code modular. 

Timeline: 5-10 days.

Commit all implementation to Github (both the colab and py files) and invite me to the repo.


Dataset:

The dataset is in csv format containing 22 fields, all in different data types, some are nullable, and others have datetime properties (so take cognizance of this).


Task:

1.	Load the csv data and answer the following questions:
-	What is the ratio of Male to Female that are skilled workers?
-	What proportion of the population who own a real estate property are also citizens?
-	What is the percentage of married people that are dependents have borrowed for business purpose, but have also defaulted on their loan?
-	Compute the total amount of Male customers with more that -100 checking balance have borrowed in the past.
-	How long does it take Female customers between the age of 30 – 50 to pay back their loan and compare that to Male customers. 
2.	Using the csv build a standard machine learning model to predict the possibility of a person to default on their loan. You can use this guide (choice of techniques is up to you):
a.	Load the data to a pandas dataframe
b.	Perform the necessary preprocessing
c.	Split the data into train and test set (you can also have a validation set if you feel it’s necessary)
d.	Train the model, and test it (we are interested in probability of person defaulting as well)
e.	Record the results using metrics such as F1-score, accuracy, precision, recall.
f.	Let us know the percent of people who accurately default
3.	Given that we have the probability of defaulting, now assuming we pick a threshold of 0.4 how many people are we going to be able to accurately give out loan too. Compare that to using a threshold of 0.6
4.	What does your observation in 3 above mean for our loan company, assuming you in a position to making decision on whether to give out loans or not. 
