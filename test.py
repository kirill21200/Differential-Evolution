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


def diff_evo(fobj, r_min, r_max, popsize, its, F=0.7, Cr=0.8, dim=1):
    pop = np.random.uniform(r_min, r_max, (popsize, dim))
    us = pop.copy()
    gen = 0

    best = min(pop, key=fobj)
    best_value = fobj(best)
    prev_best_value = best_value

    best_gen = 1

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
        if prev_best_value != best_value:
            best_gen = gen
            prev_best_value = best_value
        # results.append(pop.copy())
        # results.append
        # results.append(best_value, best_gen)
    return best_value, best_gen


#
# print(frame.describe())
#
# best_values = []
# best_gens = []
# for i in range(100):
#     data = diff_evo(rosenbrock_func, -10.0, 10.0, 100, 200, dim=10)
#     best_values.append(data[0])
#     best_gens.append(data[1])
# data = {'best_values': best_values, 'best_gens': best_gens}
# frame = pd.DataFrame(data)
# frame.to_csv('rosebrock.csv', index=False)
#
# best_values = []
# best_gens = []
# for i in range(100):
#     data = diff_evo(grivank_func, -10.0, 10.0, 100, 200, dim=10)
#     best_values.append(data[0])
#     best_gens.append(data[1])
# data = {'best_values': best_values, 'best_gens': best_gens}
# frame = pd.DataFrame(data)
# frame.to_csv('grivank.csv', index=False)


frame = pd.read_csv('rosebrock.csv')
# frame['best_values'].plot(kind='box', title='Распределение результатов 100 запусков для функции Розенброка')
# plt.show()


# # points = diff_evo(rosenbrock_func, -10.0, 10.0, 20, 100, dim=2)
# # plt.plot(range(len(points)), points)
# pyplot.show()
