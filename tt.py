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
    return np.sum(x1**2).reshape(-1, 1) + np.sum(x2**2).reshape(-1, 1) - 2*np.dot(x1, x2.T)


def cov_func_rbf(x1, x2, l2=None):
    if l2 is None:
        l2 = max(x1) - min(x2)
    return np.exp(-.5 * (1/l2) * diff_sq(x1, x2))


def visualize(X, Y):
    plt.plot(X, Y, 'x')
    plt.show()


def get_data():
    noise_var = 1e-6
    n_samples = 10
    n = 100
    X, Y = generate_points(n)
    visualize(X, Y)
    Xtest, Ytest = generate_points(n)
    Xtest.sort(axis=0)
    return noise_var, n_samples, n, X, Y, Xtest, Ytest


def solution():
    noise_var, n_samples, n, X, Y, Xtest, Ytest = get_data()
    K = cov_func_rbf(Xtest, Xtest)
    L = np.linalg.cholesky(K + noise_var*np.eye(n))
    f_prior = np.dot(L, np.random.normal(size=(n, n_samples)))
    visualize(X, f_prior)
    K = cov_func_rbf(X, X)
    L = np.linalg.cholesky(K + noise_var*np.eye(n))
    Ktest = cov_func_rbf(Xtest, Xtest)
    Lk = np.linalg.solve(L, cov_func_rbf(X, Xtest))
    mu = np.dot(Lk.T, np.linalg.solve(L, Y))
    L = np.linalg.cholesky(Ktest + noise_var ** 2 * np.eye(n) - np.dot(Lk.T, Lk))
    f_posterior = mu.reshape(-1, 1) + np.dot(L, np.random.normal(size=(n, n_samples)))
    visualize(X, f_posterior)


solution()
