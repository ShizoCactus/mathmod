import random

import numpy as np
import matplotlib.pylab as plt

import dim2gp
import custommath


def add_noise(arr, e): #Функция добавления шума
    return [x + 2 * random.random() * e - e for x in arr]


def func(x, y, a, b, c): #Моделируемая функция
    t1 = np.subtract(y, c)
    t2 = np.square(t1)
    t3 = np.multiply(x, a)
    t4 = np.multiply(y, b)
    t5 = np.sin(t3 + t4)
    z = np.multiply(t2, t5)
    return z


def exponential_cov(x, y, gamma, theta): #Функция ядра
    return gamma * np.exp(-0.5 * theta * np.subtract.outer(x, y)**2)


def arr_sum(arr1, arr2): #Попарное сложение элементов массива
    return [arr1[i] + arr2[i] for i in range(min(len(arr1), len(arr2)))]


def exponential_cov_2(x1, x2, y1, y2, gamma, theta): #Функция ядра для 2 переменных
    return np.add(gamma * np.exp(-0.5 * theta * np.subtract.outer(x1, x2)**2),
                  gamma * np.exp(-0.5 * theta * np.subtract.outer(y1, y2)**2))


class GP2:
    def __init__(self):
        self.x = [0]
        self.y = [0]
        self.z = [[0]]
        self.gamma = 1
        self.theta = 10
        self.sigma = exponential_cov_2(self.x, self.x, self.y, self.y, self.gamma, self.theta)

    def predict(self, x, y, data, kernel_2, gamma, theta, sigma, t): #Функция предсказания
        k = [kernel_2(x, z, y, z, gamma, theta) for z in data]
        sigma_inv = np.linalg.inv(sigma)
        z_pred = custommath.mult(custommath.mult(k, sigma_inv), t)
        sigma_new = kernel_2(x, x, y, y, gamma, theta) - custommath.mult(custommath.mult(k, sigma_inv), k)
        return z_pred, sigma_new

    def work(self):
        #Параметры
        a = 2
        b = 3
        c = 1
        n = 100
        ampl = 5
        gamma = 1
        theta = 10
        x_graph = np.linspace(-5, 5, 1000)
        x, y = np.meshgrid(np.linspace(-ampl, ampl, n), np.linspace(-5, 5, n))
        x_noisy = add_noise(x, ampl / 20)
        y_noisy = add_noise(y, ampl / 20)
        z = func(x_noisy, y_noisy, 2, 5, 4)
        self.x += x
        self.y += y
        self.z += z
        fig2 = plt.figure()
        ax = fig2.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(self.x, self.y, self.z)
        plt.show()
        sigma = exponential_cov_2(self.x, self.x, self.y, self.y, gamma, theta)
        coord_y = 2
        gp_1dim = dim2gp.GP()
        gp_1dim.work(a, b, c, coord_y, func)


GP2().work()
