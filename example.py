import random

import numpy as np
import matplotlib.pylab as plt

def f(x, y, a, b, c):
    return (y - c) ** 2 * np.sin(a * x + b * y)


def example_f(x):
    return x ** 2 * np.exp(-x) * np.cos(3 * x)


def generate_points(n=10):
    np.random.seed(777)
    x = np.random.random_sample(n) * (-10)
    y = example_f(x)
    return x, y


def generate_points_with_noise(n=10, noise_variance=1e-6):
    X, y = generate_points(n)
    y += np.random.randn(n, 1) * noise_variance ** 0.5
    return X, y


def diff_sq(x1, x2):
    return (x1 - x2) ** 2


def cov_func_rbf(x1, x2, l2=0.05):
    return np.exp(-0.5 * (1/l2) * diff_sq(x1, x2))


def cov_matrix(x1, x2):
    lst = []
    for i in range(len(x1)):
        arr = np.array([cov_func_rbf(x1[i], x2[j]) for j in range(len(x2))])
        lst.append(arr)
    return np.array(lst)


def visualize(X, Y):
    plt.plot(X, Y, 'x')
    plt.show()


def get_data():
    noise_var = 1e-6
    n_samples = 10
    n = 10
    m = np.zeros(n)
    X, Y = generate_points(n)
    Xtest, Ytest = generate_points(n)
    Xtest.sort(axis=0)
    return m, noise_var, n_samples, n, X, Y, Xtest, Ytest


def solution():
    m, noise_var, n_samples, n, X, Y, Xtest, Ytest = get_data()
    K = cov_func_rbf(X, X)
    print(K)
    f_prior = Y
    visualize(X, f_prior)
    m = m + cov_func_rbf(X, Xtest) * cov_func_rbf(Xtest, Xtest).reshape((-1, 1))
    Ktest = cov_func_rbf(Xtest, Xtest)
    # mu = np.dot(K.T, np.linalg.solve(K, Y))
    # f_posterior = mu.reshape(-1, 1) + np.dot(K, np.random.normal(size=(n, n_samples)))
    # visualize(X, f_posterior)


solution()
