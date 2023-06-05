import random

import numpy as np
import matplotlib.pylab as plt

import custommath


def exponential_cov(x, y, gamma, theta): #Функция ядра
    return gamma * np.exp(-0.5 * theta * np.subtract.outer(x, y)**2)


def example_func(x, a, b, c, d): #Моделируемая функция
    t1 = np.subtract(d, c)
    t2 = np.square(t1)
    t3 = np.multiply(x, a)
    t4 = np.multiply(x, b)
    t5 = np.sin(t3 + t4)
    y = np.multiply(t2, t5)
    return y


class GP:
    def __init__(self):
        self.x = [0]
        self.y = [0]
        self.gamma = 1
        self.theta = 10
        self.sigma = exponential_cov(self.x, self.x, self.gamma, self.theta)

    def predict(self, x, data, kernel, gamma, theta, sigma, t): #Предсказание среднего и погрешности
        k = [kernel(x, y, gamma, theta) for y in data]
        sigma_inv = custommath.inverse(sigma)
        y_pred = custommath.mult(custommath.mult(k, sigma_inv), t)
        sigma_new = kernel(x, x, gamma, theta) - custommath.mult(custommath.mult(k, sigma_inv), k)
        return y_pred, sigma_new

    def work(self, a, b, c, d, fn=example_func): #Построение графика на основе предсказания
        gamma = 1         #Гиперпараметры гамма и дзета
        theta = 10
        noise = 0.1        #Шум
        x_graph = np.linspace(-5, 5, 1000)
        x_new = [random.random() * 10 - 5 for i in range(15)] #Случайно выбранные точки
        y_new = fn(x_new, 2, 5, 4, 1)  #Значение функции в этих точках
        y_new = [y + 2 * noise * random.random() - noise for y in y_new] #Добавление шума
        self.x += x_new
        self.y += y_new
        sigma = exponential_cov(self.x, self.x, gamma, theta)
        # Предсказание среднего и отклонения
        predictions = [self.predict(i, self.x, exponential_cov, gamma, theta, sigma, self.y) for i in x_graph]
        y_graph, sigmas = np.transpose(predictions)
        y_graph = np.array([t + random.random() * noise for t in y_graph])
        sigmas = np.array([t + random.random() * noise for t in sigmas])
        # Отрисовка графика
        plt.errorbar(x_graph, y_graph, yerr=abs(sigmas), capsize=0)
        plt.plot(self.x, self.y, "ro")
        plt.show()
