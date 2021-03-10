import pandas
import math


# Data from https://www.kaggle.com/c/titanic/data


def get_first_name(source_name: str) -> str:
    # name = row["Name"]
    name = source_name
    if ',' in name:
        name = name.split(',', 2)[1].strip()
    # name = name.replace("Mr.", "").strip()
    name = name.replace("Mrs.", "").strip()
    name = name.replace("Miss.", "").strip()
    # name = name.replace("Master.", "").strip()
    if '(' in name:
        name = name[name.rfind('(') + 1:-1]
    name = name.split(' ')[0]
    return name


data = pandas.read_csv('source_data/titanic.csv', index_col='PassengerId')

sex = dict(data['Sex'].value_counts())
print(sex['male'], sex['female'])

total = len(data)

survived = float(dict(data["Survived"].value_counts())[1]) / total  # [1] - survived, not the first
print(f'{survived:.2f}')

first_class = float(dict(data['Pclass'].value_counts())[1]) / total  # [1] - first class, not the first item
print(f'{first_class:.2f}')

average = data['Age'].mean()
median = data['Age'].median()
print(f'{average:.2f} {median:.2f}')

cr = data.corr().loc["SibSp", "Parch"]
print(f'{cr:.2f}')

female_names = data.loc[data["Sex"] == 'female']
# female_names = female_names.loc[~female_names["Name"].str.contains("Mr.") & ~female_names["Name"].str.contains("Master.")]

# fn = list(map(get_first_name, list(female_names["Name"])))
data["FName"] = female_names["Name"].apply(lambda row: get_first_name(row))
female_names_rate = data["FName"].value_counts(sort=True)

print(female_names_rate.head(n=5))
