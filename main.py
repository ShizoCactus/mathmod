import numpy as np
import matplotlib.pyplot as plt
import custom_math


def transform_array_to_matrix(arr):
    n = int(len(arr) ** 0.5)
    mtrx = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            mtrx[i][j] = arr[i * n + j]

    return mtrx


def generate_data(a, b, c, n=20, noise_variance=1e-6): #Генерация случайных стартовых данных
    np.random.seed(666)
    x1 = np.random.uniform(-3.0, 3.0, n)
    x1.sort()
    x2 = np.random.uniform(-3.0, 3.0, n)
    x2.sort()
    y = (x2 - c) ** 2 * np.sin(a * x1 + b * x2) + np.random.randn() * noise_variance**0.5
    return x1, x2, y


def exponential_cov(x, y, sigma, l2): #Функция ядра
    sqdist = np.sum(x**2, 1).reshape(-1,1) + np.sum(y**2, 1) - 2*np.dot(x, y.T)
    return sigma ** 2 * np.exp(-0.5 * (1/l2) * sqdist)


def predict(X, Xtest, sigma=1, l2=0.1, noise_var=1e-6): #Основной алгоритм
    N, n = len(X), len(Xtest)
    K = exponential_cov(X, X, sigma, l2)
    L = custom_math.my_cholesky(K + noise_var*np.eye(N))
    Lk = custom_math.my_solve(L, exponential_cov(X, Xtest, sigma, l2))
    mu = custom_math.my_dot2(custom_math.my_transpose(Lk), custom_math.my_solve(L, y))
    K_ = exponential_cov(Xtest, Xtest, sigma, l2)
    sd = np.sqrt(custom_math.get_diagonal(K_) - np.sum(Lk**2, axis=0))
    return (mu, sd)


sigma = 3 # Гиперпараметры
l2 = 0.2

a = 2 #Параметры
b = 3
c = 1
x1, x2, y = generate_data(a, b, c)
X = np.column_stack((x1, x2))  # Объединение двух переменных в одну матрицу
x1_graph, x2_graph, y_graph = generate_data(a, b, c, 1000)
X_graph = np.column_stack((x1_graph, x2_graph))  # Объединение двух переменных в одну матрицу
X_graph.sort(axis=0)
mu, sd = predict(X, X_graph, sigma, l2)


# Визуализация данных 2d
summa = np.array([t[0] + t[1] for t in X])
summa_graph = np.array([t[0] + t[1] for t in X_graph])
y_min = np.array(mu - sd)
y_max = np.array(mu + sd)
plt.plot(x2, y, 'x')
plt.plot(x2_graph, mu, label='Mean')
plt.fill_between(x2_graph, y_min, y_max, alpha=0.3)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


# Визуализация данных 3d

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(transform_array_to_matrix(x1_graph[271:1000]), transform_array_to_matrix(x2_graph[271:1000]), transform_array_to_matrix(y_graph[271:1000]))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(transform_array_to_matrix(x1_graph[100:1000]), transform_array_to_matrix(x2_graph[100:1000]), transform_array_to_matrix(mu[100:1000]))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(transform_array_to_matrix(x1_graph[100:1000]), transform_array_to_matrix(x2_graph[100:1000]), transform_array_to_matrix(mu[100:1000] - sd[100:1000]))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(transform_array_to_matrix(x1_graph[100:1000]), transform_array_to_matrix(x2_graph[100:1000]), transform_array_to_matrix(mu[100:1000] + sd[100:1000]))
plt.show()


def check_accuracy(left, right):
    ans = 0
    for i in range(left, right):
        x1_g = x1_graph[i]
        x2_g = x2_graph[i]
        y_g = y_graph[i]
        m = mu[i]
        s = sd[i]
        b = m - s <= y_g <= m + s
        if b:
            ans += 1
        # print(i)
        # print('x1=', x1_g, 'x2=', x2_g, 'f=', y_g, 'm=', m, 's=', s, b)
    print(ans / (right - left))


check_accuracy(0, 1000)
check_accuracy(250, 1000)
check_accuracy(300, 1000)
