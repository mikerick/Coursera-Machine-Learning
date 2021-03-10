import numpy

"""
loc: среднее нормального распределения (в нашем случае 1)
scale: стандартное отклонение нормального распределения (в нашем случае 10)
size: размер матрицы (в нашем случае (1000, 50))
"""
X = numpy.random.normal(loc=1, scale=10, size=(1000, 50))
print(X)

# Нормировка матрицы

m = numpy.mean(X, axis=0)
std = numpy.std(X, axis=0)
X_norm = ((X - m) / std)
print(X_norm)

Z = numpy.array(
    [
        [4, 5, 0],
        [1, 9, 3],
        [5, 1, 1],
        [3, 3, 3],
        [9, 9, 9],
        [4, 7, 1]
    ]
)

r = numpy.sum(Z, axis=1)
print(r)

A = numpy.eye(3)
B = numpy.eye(3)
print(A)
print(B)
AB = numpy.vstack((A, B))
print(AB)

AB2 = numpy.hstack((A, B))
print(AB2)
