# import sklearn
from sklearn import model_selection as ms
from sklearn import neighbors as nb
from sklearn import preprocessing as pp
import numpy as np
import pandas

# sklearn.neighbors.KNeighborsClassifier() - метод k ближайших соседей

# 1. набор разбиенй на обучение и валидацию sklearn.model_selection.KFold
# 2. вычислить качество на разбиении при помощи sklearn.model_selection.cross_val_score()


"""
Дополнительные ссылки для выполнения задания:
https://habr.com/ru/company/ods/blog/322534/#klass-kneighborsclassifier-v-scikit-learn
https://tproger.ru/translations/scikit-learn-in-python/
https://habr.com/ru/company/mlclass/blog/247751/
http://zabaykin.ru/?p=667
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html?highlight=kfold#sklearn.model_selection.KFold
https://www.machinelearningmastery.ru/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02/
https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array
https://codecamp.ru/blog/cross-validation-k-fold/
"""

"""
1. Загрузите выборку Wine по адресу https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data 
    (файл также приложен к этому заданию)
"""
# OK!
source_data_path = './source_data/wine.data'
"""
2. Извлеките из данных признаки и классы. Класс записан в первом столбце (три варианта), 
    признаки — в столбцах со второго по последний. Более подробно о сути признаков можно прочитать 
    по адресу https://archive.ics.uci.edu/ml/datasets/Wine (см. также файл wine.names, приложенный к заданию)
    
    P.S. Можно использовать wine.names из задания, что было проигнорировано за ненадобностью. 
"""
data = pandas.read_csv(source_data_path,
                       names=['class', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12',
                              'f13'])
"""
3. Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold). 
    Создайте генератор разбиений, который перемешивает выборку перед формированием блоков (shuffle=True). 
    Для воспроизводимости результата, создавайте генератор KFold с фиксированным параметром 
    random_state=42. В качестве меры качества используйте долю верных ответов (accuracy).
"""
partitioner = ms.KFold(n_splits=5, shuffle=True, random_state=42)
"""
4. Найдите точность классификации на кросс-валидации для метода k ближайших соседей 
    (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50. При каком k получилось 
    оптимальное качество? Чему оно равно (число в интервале от 0 до 1)? Данные результаты 
    и будут ответами на вопросы 1 и 2.
"""

kresults = list()  # index = neighbors - 1
for k in range(50):
    classifier = nb.KNeighborsClassifier(n_neighbors=k + 1)
    scores = ms.cross_val_score(
        estimator=classifier,
        X=data.drop(['class'], axis=1).to_numpy(),
        y=data['class'].to_numpy(),
        cv=partitioner)
    kresults.append(scores.mean())

print(max(kresults), kresults.index(max(kresults)) + 1)

with open('./result_data/task_03.1.txt', 'w') as f:
    f.write(str(kresults.index(max(kresults)) + 1))

with open('./result_data/task_03.2.txt', 'w') as f:
    f.write(f"{max(kresults):0.2f}")

"""
5. Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale. 
    Снова найдите оптимальное k на кросс-валидации.
"""

scaled_data = pp.scale(data.drop(['class'], axis=1))
kresults = list()
for k in range(50):
    classifier = nb.KNeighborsClassifier(n_neighbors=k + 1)
    scores = ms.cross_val_score(
        estimator=classifier,
        X=scaled_data,
        y=data['class'].to_numpy(),
        cv=partitioner
    )
    kresults.append(scores.mean())

"""
6. Какое значение k получилось оптимальным после приведения признаков к одному масштабу? 
    Приведите ответы на вопросы 3 и 4. Помогло ли масштабирование признаков?
    
    Масштабирование помогло.
"""

print(max(kresults), kresults.index(max(kresults)) + 1)

with open('./result_data/task_03.3.txt', 'w') as f:
    f.write(str(kresults.index(max(kresults)) + 1))

with open('./result_data/task_03.4.txt', 'w') as f:
    f.write(f"{max(kresults):0.2f}")
