# import sklearn
from sklearn import datasets as ds
from sklearn import model_selection as ms
from sklearn import neighbors as nb
from sklearn import preprocessing as pp
import numpy
import pandas

"""
Ссылки для выполнения задания:
https://pythonist-ru.turbopages.org/pythonist.ru/s/funkczii-numpy-linspace-i-numpy-logspace/
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

"""

""" Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston().
    Результатом вызова данной функции является объект, у которого признаки записаны в
    поле data, а целевой вектор — в поле target."""

boston_data = ds.load_boston()

""" Приведите признаки в выборке к одному масштабу при помощи функции
    sklearn.preprocessing.scale."""

pp.scale(boston_data['data'])

""" Переберите разные варианты параметра метрики p по сетке от 1 до 10
    с таким шагом, чтобы всего было протестировано 200 вариантов 
    (используйте функцию numpy.linspace). Используйте KNeighborsRegressor с 
    n_neighbors=5 и weights='distance' — данный параметр добавляет в алгоритм веса, 
    зависящие от расстояния до ближайших соседей. В качестве метрики качества 
    используйте среднеквадратичную ошибку (параметр scoring='mean_squared_error' 
    у cross_val_score; при использовании библиотеки scikit-learn версии 0.18.1 
    и выше необходимо указывать scoring='neg_mean_squared_error').  
    Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации 
    по 5 блокам с random_state = 42, не забудьте включить перемешивание выборки 
    (shuffle=True)."""

# p = 1
folder = ms.KFold(n_splits=5, random_state=42, shuffle=True)
scores = dict()
for p in numpy.linspace(start=1.0, stop=10.0, num=200, endpoint=True):
    regressor = nb.KNeighborsRegressor(
        n_neighbors=5,
        metric='minkowski',
        weights='distance',
        p=p
    )
    cvs = ms.cross_val_score(
        estimator=regressor,
        X=boston_data['data'],
        y=boston_data['target'],
        scoring='neg_mean_squared_error',
        cv=folder
    ).mean()
    scores[p] = cvs
    print(f"Metric value: {p}, CVS: {cvs:0.2f}")

""" Определите, при каком p качество на кросс-валидации оказалось оптимальным. 
    Обратите внимание, что cross_val_score возвращает массив показателей качества 
    по блокам; необходимо максимизировать среднее этих показателей. Это значение 
    параметра и будет ответом на задачу."""

sorted_scores = sorted(scores.items(), reverse=True, key=lambda x: x[1])

with open('./result_data/task_04.1.txt', 'w') as f:
    f.write(f"{sorted_scores[0][0]:0.2f}")
