import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization


# функция с 2-мя неизвестными (x и y) и 3-мя параметрами (a, b и c)
def f(x, y, a, b, c):
    return (y - c) ** 2 * np.sin(a * x + b * y)


# визуализация фунции с заданными параметрами, использует библиотеку matplotlib
def visualize(a, b, c):
    start = -5
    end = 5
    count = 1000
    x_val = np.linspace(start, end, count)
    y_val = np.linspace(start, end, count)
    x, y = np.meshgrid(x_val, y_val) # массивы точек x, y по которым построится график на промежутке от start до end
    z = f(x, y, a, b, c) # значение функции в этих точках для заданных параметров a, b, c
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='inferno')
    ax.legend()
    fig.show()
    ...


# сбор значений функции с заданными параметрами для ограниченного числа точек, по которым впоследствии
# будут восстанавливаться параметры
def collect_data(a, b, c):
    start = -5
    end = 5
    count = 20
    x_val = np.linspace(start, end, count)
    y_val = np.linspace(start, end, count)
    x, y = np.meshgrid(x_val, y_val) # массивы точек x, y на промежутке от start до end
    z = f(x, y, a, b, c) # значение функции в этих точках для заданных параметров a, b, c
    return z


# выводит на экран значения двумерного массива, округляя значения в них, для удобства предстваления
def print_data(matrix):
    for arr in matrix:
        for element in arr:
            print(round(element, 2), end=' ')
        print()


def main():

    def set_params(a, b, c):
        return lambda x, y: f(x, y, a, b, c)

    func = set_params(1, 2, 3)

    pbounds = {'x': (2, 4), 'y': (-3, 3)}
    optimizer = BayesianOptimization(
        f=func,
        pbounds=pbounds,
        random_state=1
    )
    optimizer.maximize()
    print(optimizer.max)
    # a = 1
    # b = 2
    # c = 3
    # visualize(a, b, c)
    # data = collect_data(a, b, c)
    # print_data(data)


main()
