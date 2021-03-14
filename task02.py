import pandas
import sklearn
import sklearn.tree as stree
import math


# Data from https://www.kaggle.com/c/titanic/data


data = pandas.read_csv('source_data/titanic.csv', index_col='PassengerId')

data["NSex"] = data["Sex"].apply(lambda x: 0 if x == 'female' else 1)
data = data[["Pclass", "Fare", "Age", "NSex", "Survived"]].dropna()

clf = stree.DecisionTreeClassifier(random_state=241)
clf.fit(data[["Pclass", "Fare", "Age", "NSex"]], data[["Survived"]])

print("ok!")