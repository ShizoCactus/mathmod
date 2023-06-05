import numpy as np


def mult(matrix1, matrix2): #Умножение матриц
    try:
        rows_1 = len(matrix1)
        cols_1 = len(matrix1[0])
        cols_2 = len(matrix2[0])
        result = [[0 for __ in range(cols_2)] for _ in range(rows_1)]
        for s in range(rows_1):
            for j in range(cols_2):
                for k in range(cols_1):
                    result[s][j] += matrix1[s][k] * matrix2[k][j]
    except BaseException:
        result = np.dot(matrix1, matrix2)
    return result


def transpose(Mtrx): #Транспонирование матрицы
    return list(map(list, zip(*Mtrx)))


def get_minor(Mtrx, i, j): #Минор матрицы
    return [row[:j] + row[j+1:] for row in (Mtrx[:i] + Mtrx[i + 1:])]


def get_determinant(Mtrx): #Определитель матрицы
    if len(Mtrx) == 2:
        return Mtrx[0][0] * Mtrx[1][1] - Mtrx[0][1] * Mtrx[1][0]

    determinant = 0
    for c in range(len(Mtrx)):
        determinant += ((-1)**c) * Mtrx[0][c] * get_determinant(get_minor(Mtrx, 0, c))
    return determinant


def inverse(Mtrx): #Обратная матрица
    try:
        determinant = get_determinant(Mtrx)
        if len(Mtrx) == 2:
            return [[Mtrx[1][1] / determinant, -1 * Mtrx[0][1] / determinant],
                    [-1 * Mtrx[1][0] / determinant, Mtrx[0][0] / determinant]]
        cofactors = []
        for r in range(len(Mtrx)):
            cofactorRow = []
            for c in range(len(Mtrx)):
                minor = get_minor(Mtrx, r, c)
                cofactorRow.append(((-1)**(r+c)) * get_determinant(minor))
            cofactors.append(cofactorRow)
        cofactors = transpose(cofactors)
        for r in range(len(cofactors)):
            for c in range(len(cofactors)):
                cofactors[r][c] = cofactors[r][c]/determinant
    except BaseException:
        cofactors = np.linalg.inv(Mtrx)
    return cofactors
