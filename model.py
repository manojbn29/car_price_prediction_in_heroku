import pandas as pd

df = pd.read_csv('car data.csv')

print(df.head())

# Dropping car_name
final_dataset = df.drop(['Car_Name'], axis = 1)

final_dataset['Current_Year'] =2021

final_dataset['no_of_years'] = final_dataset['Current_Year'] - final_dataset['Year']

final_dataset.drop(['Year', 'Current_Year' ], axis = 1, inplace = True)

final_dataset = pd.get_dummies(final_dataset, drop_first = True)

print(final_dataset.head())

X = final_dataset.iloc[:, 1:]

y = final_dataset.iloc[:, 0]

#Doing train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Doing Ridge regression

from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha = 0.001)   # I checked alpha value by trail and error method

ridge_model.fit(X_train, y_train)

y_predict = ridge_model.predict(X_test)

from sklearn.metrics import r2_score
accuracy = r2_score(y_test, y_predict)
print(accuracy)


# Importing to pickle file
import pickle
# open a file, where you ant to store the data
file = open('ridge_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(ridge_model, file)