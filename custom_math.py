import numpy as np


def my_dot(a, b): # Скалярное произведение векторов
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]

    return result


def my_dot2(a, b): # Умножение матрицы на вектор
    result = [0] * len(a)

    for i in range(len(a)):
        for j in range(len(b)):
            result[i] += a[i][j] * b[j]

    return result


def my_transpose(matrix): # Транспонирование матрицы
    rows = len(matrix)
    cols = len(matrix[0])

    transposed_matrix = np.zeros((cols, rows))

    for i in range(rows):
        for j in range(cols):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix


def my_cholesky(A): # Разложение Холецкого
    n = A.shape[0]
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - np.sum(L[i][:j] ** 2))
            else:
                L[i][j] = (A[i][j] - my_dot(L[i][:j], L[j][:j])) / L[j][j]

    return L


def my_solve(A, b): # Решение СЛАУ
    try:
        n = A.shape[0]
        x = np.zeros(n)

        for i in range(n-1, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    except:
        x = np.linalg.solve(A, b)

    return x


def get_diagonal(matrix): # Выделение диагонали квадратной матрицы
    rows = len(matrix)
    cols = len(matrix[0])

    if rows != cols:
        raise ValueError("Input matrix is not square")

    diagonal = [matrix[i][i] for i in range(rows)]

    return diagonal
