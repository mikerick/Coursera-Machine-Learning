"""
Задача на линейные методы классификации
"""
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
import pandas

"""
Ссылки для выполнения задания:
https://thispointer.com/python-numpy-select-rows-columns-by-index-from-a-2d-ndarray-multi-dimension/
https://www.kite.com/python/answers/how-to-select-columns-from-a-pandas-%60dataframe%60-by-index-in-python#:~:text=B%2C%20dtype%3A%20int64-,Use%20DataFrame.,indices%20a%20up%20to%20b%20.
https://www.kite.com/python/answers/how-to-create-pandas-dataframe-from-a-numpy-array-in-python
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
https://stackoverflow.com/questions/35723472/how-to-use-sklearn-fit-transform-with-pandas-and-return-dataframe-instead-of-num
https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array
https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points
https://github.com/tyz910/hse-shad-ml/blob/master/05-statement-linear/Solution.ipynb
"""

"""
1. Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv.
Целевая переменная записана в первом столбце, признаки — во втором и третьем.
"""
# Удобно сразу разбивать данные на X, y (как аргументы функций)
train_data = pandas.read_csv('./source_data/perceptron-train.csv', names=[0, 1, 2])
X_train = train_data.loc[:, 1:]
y_train = train_data[0]
test_data = pandas.read_csv('./source_data/perceptron-test.csv', names=[0, 1, 2])
X_test = test_data.loc[:, 1:]
y_test = test_data[0]

"""
2. Обучите персептрон со стандартными параметрами и random_state=241.
"""
# Параметры max_iter и tol делают ответ правильным. Без них не заработает
# Из документации:
# tol float, default=1e-3
# The stopping criterion. If it is not None, the iterations will stop when (loss > previous_loss - tol).
# max_iterint, default=1000
# The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit method.
lin_model = Perceptron(random_state=241, max_iter=5, tol=None)
lin_model.fit(X_train, y_train)

"""
3. Подсчитайте качество (долю правильно классифицированных объектов, accuracy) полученного классификатора на тестовой выборке.
"""
accuracy = accuracy_score(y_test, lin_model.predict(X_test))

"""
4. Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.
"""
scaler = StandardScaler()
# Сначала обучаем нормализатор, потом нормализуем тестовую выборку
# Добавил ссылку, как пересобрать выборки, схожие с исходными, но, как оказалось, это вовсе не требуется
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

"""
5. Обучите персептрон на новой выборке. Найдите долю правильных ответов на тестовой выборке.
"""

lin_model.fit(X_scaled_train, y_train)
scaled_accuracy = accuracy_score(y_test, lin_model.predict(X_scaled_test))

"""
6. Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее. 
Это число и будет ответом на задание.
"""
print(f"{round(scaled_accuracy - accuracy, 3)}")
with open('./result_data/task_05.01.txt', 'w') as of:
    of.write(f'{round(scaled_accuracy - accuracy, 3)}')
