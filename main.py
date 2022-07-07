import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.animation import ArtistAnimation
from numpy import e, exp, cos, pi, sqrt, arange, meshgrid, square, prod


def eckley_func(x):
    d = len(x)
    sum1 = 0
    sum2 = 0
    for i in range(d):
        sum1 += square(x[i])
        sum2 += cos(2 * pi * x[i])
    return -20 * exp(-0.2 * sqrt(sum1 / d)) - exp(sum2 / d) + 20 + exp(1)


def rosenbrock_func(x):
    sum = 0
    for i in range(len(x) - 1):
        sum += 100 * square(x[i + 1] - square(x[i])) + square(x[i] - 1)
    return sum


def grivank_func(x):
    sum = 0
    prod = 1
    for i in range(len(x)):
        sum += square(x[i]) / 4000
        prod *= cos(x[i] / sqrt(i + 1))
    return sum - prod + 1


def de(fobj, r_min, r_max, popsize, its, F=0.7, Cr=0.8, dim=1):
    pop = np.random.uniform(r_min, r_max, (popsize, dim))
    us = pop.copy()
    gen = 0

    best = min(pop, key=fobj)
    best_value = fobj(best)

    results = []

    while (gen < its and best_value != 0):
        gen += 1

        for i in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), r_min, r_max)

            J = np.random.randint(dim)

            for j in range(dim):
                rand_float = np.random.random()
                if rand_float <= Cr or J == j:
                    us[i][j] = mutant[j]
                else:
                    us[i][j] = pop[i][j]

        for i in range(popsize):
            pop[i] = min((pop[i], us[i]), key=fobj)

        best = min(pop, key=fobj)
        best_value = fobj(best)
        # results.append(pop.copy())
        results.append(best_value)

    return results


def draw_3d_points(units, func, axis):
    res = []
    xaxis = []
    yaxis = []
    for i in range(len(units)):
        res.append(func(units[i]))
        xaxis.append(units[i][0])
        yaxis.append(units[i][1])

    axis.scatter(xaxis, yaxis, res, color='r')


def drow_3d_graph(func, r_min, r_max, axis):
    xaxis = arange(r_min, r_max, 0.1)
    yaxis = arange(r_min, r_max, 0.1)
    x, y = meshgrid(xaxis, yaxis)

    res = [[0] * len(x) for i in range(len(y))]
    for i in range(len(x)):
        for j in range(len(y)):
            unit = np.array([x[i][j], y[i][j]])
            res[i][j] = func(unit)
    results = np.array(res)

    axis.plot_wireframe(x, y, results, alpha=0.1)
    # axis.plot_surface(x, y, results, cmap='jet')


def draw_2d_graph(func, r_min, r_max, axis):
    xaxis = arange(r_min, r_max, 0.1)
    yaxis = []
    for i in range(len(xaxis)):
        yaxis.append(func([xaxis[i]]))
    plt.plot(xaxis, yaxis)


def draw_2d_points(units, func, axis):
    res = []
    xaxis = []
    for i in range(len(units)):
        res.append(func(units[i]))
        xaxis.append(units[i][0])
    plt.scatter(xaxis, res, color='r')


def do_2d_best_value_animation(units, func, axis, figure):
    frames = []
    for u in units:
        yaxis = func(u)
        point = axis.scatter(u[0], yaxis, color='r')
        frames.append([point])

    animation = ArtistAnimation(
        figure,  # фигура, где отображается анимация
        frames,  # кадры
        interval=100,  # задержка между кадрами в мс
        blit=True,  # использовать ли двойную буферизацию
        repeat=False)
    animation.save('best_values.gif', writer='pillow')


def do_2d_pop_animation(pops, func, axis, figure):
    frames = []
    for pop in pops:
        xaxis = []
        yaxis = []
        for unit in pop:
            xaxis.append(unit[0])
            yaxis.append(func([unit[0]]))
        points = axis.scatter(xaxis, yaxis, color='r')
        frames.append([points])

    animation = ArtistAnimation(
        figure,  # фигура, где отображается анимация
        frames,  # кадры
        interval=100,  # задержка между кадрами в мс
        blit=True,  # использовать ли двойную буферизацию
        repeat=False)
    animation.save('2d_pops.gif', writer='pillow')


def do_3d_pop_animation(pops, func, axis, figure):
    frames = []
    for pop in pops:
        xaxis = []
        yaxis = []
        res = []
        for unit in pop:
            xaxis.append(unit[0])
            yaxis.append(unit[1])
            res.append(func(unit))
        points = axis.scatter(xaxis, yaxis, res, color='r')
        frames.append([points])

    animation = ArtistAnimation(
        figure,  # фигура, где отображается анимация
        frames,  # кадры
        interval=100,  # задержка между кадрами в мс
        blit=True,  # использовать ли двойную буферизацию
        repeat=False)
    animation.save('3d_pops.gif', writer='pillow')


# units = de(eckley_func, -10.0, 10.0, 20, 40, dim=2)
# drow_3d_graph(rosenbrock_func, -10.0, 10.0)
# draw_3d_points(units, eckley_func)

# units = de(eckley_func, -10.0, 10.0, 5, 1, dim=1)
# print(units)
# draw_2d_graph(eckley_func, -10.0, 10.0)
# draw_2d_points(units, eckley_func)

def pop_anim_2d(func, name):
    figure = pyplot.figure()
    axis = figure.add_subplot()
    figure.suptitle('Изменение популяции для 2-мерной функции ' + name)
    axis.set_xlabel('Параметр идивида')
    axis.set_ylabel('Приспособленность')

    draw_2d_graph(func, -10.0, 10.0, axis)
    pops = de(func, -10.0, 10.0, 20, 40, dim=1)
    do_2d_pop_animation(pops, func, axis, figure)


def pop_anim_3d(func, name):
    figure = pyplot.figure()
    axis = figure.add_subplot(projection='3d')
    figure.suptitle('Изменение популяции для 3-мерной функции ' + name)
    axis.set_xlabel('Первый параметр идивида')
    axis.set_ylabel('Второй параметр идивида')
    axis.set_zlabel('Приспособленность')

    pops = de(func, -10.0, 10.0, 20, 100, dim=2)
    drow_3d_graph(func, -10.0, 10.0, axis)
    do_3d_pop_animation(pops, func, axis, figure)


# pop_anim_2d(grivank_func, 'Гринвака')
# pop_anim_3d(rosenbrock_func, 'Розенброка')

figure = pyplot.figure()
axis = figure.add_subplot()
def best_values_graph(func, name, dim, popsize, its):

    figure.suptitle(f"Приспособленность {dim}-мерной функции {name} при популяции {popsize}")
    axis.set_xlabel('Поколение')
    axis.set_ylabel('Приспособленность')

    best_values = de(func, -10.0, 10.0, popsize, its, dim=dim)
    xaxis = range(len(best_values))
    axis.plot(xaxis, best_values)


best_values_graph(grivank_func, 'Гринвака', 10, 100, 200)
best_values_graph(grivank_func, 'Гринвака', 10, 100, 200)
best_values_graph(grivank_func, 'Гринвака', 10, 100, 200)



# u = de(eckley_func, -10.0, 10.0, 20, 200, dim=10)
# plt.plot(range(len(u)), u)
# u = de(eckley_func, -10.0, 10.0, 20, 200, dim=10)
# plt.plot(range(len(u)), u)
# u = de(eckley_func, -10.0, 10.0, 20, 200, dim=10)
# plt.plot(range(len(u)), u)


pyplot.show()
