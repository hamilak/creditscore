import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import scale, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score as cvs

# loading csv file
df = pd.read_csv('credit.csv')

# check for missing values
print(df.isnull())
print(df.isna().sum()/len(df)*100)

# dropping columns
data = df.drop('telephone', axis=1)

# filling missing values
object_type = data.select_dtypes(include=[object])
object_column = object_type.columns
data[object_column] = data[object_column].fillna(data[object_column].mode().iloc[0])

numeric = data.select_dtypes(include=np.number)
numeric_column = numeric.columns
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit_transform(data[numeric_column])
data[numeric_column] = imp_mean.transform(data[numeric_column])
# print(data.isnull())
# print(data.isna().sum()/len(df)*100)

# normalization
data['amount'] = MinMaxScaler().fit_transform(np.array(data['amount']).reshape(-1, 1))
# print(data['amount'])

# splitting the data into test and train sets
y = data.pop('default')
X = data
le = LabelEncoder()
for obj in object_column:
    X[obj] = le.fit_transform(X[obj].astype(str))
# print(X.info())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# SVM classifier
svm_clf = SVC(kernel='linear', random_state=0)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)
accuracy_svm = metrics.accuracy_score(y_test, y_pred_svm)
# print(accuracy_svm)
cm = confusion_matrix(y_test, y_pred_svm)
# print(cm)

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
# print(y_pred_rfc)
accuracy = metrics.accuracy_score(y_test, y_pred_rfc)
print(f'Accuracy of the model:{accuracy}')

# Logistic Regression Classifier
# lr = LogisticRegression(solver='liblinear', random_state=0)
# lr.fit(X_train, y_train)
# log_pred = lr.predict(X_test)
# scores_logistic=cvs(LogisticRegression(), X_train, y_train, cv=3)
# # train_result = lr.fit(X_train, y_train)
# print(scores_logistic)


def count_default():
    y_df = pd.DataFrame(y, columns=['default'])
    length = len(y_df.query('default == 1'))
    print(f'the length is {length}')


count_default(y_test)

