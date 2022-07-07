import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy import e, exp, cos, pi, sqrt, arange, meshgrid, square, prod


def f(x):
    d = len(x)
    sum1 = 0
    sum2 = 0
    for i in range(d):
        sum1 += square(x[i])
        sum2 += cos(2 * pi * x[i])
    return -20 * exp(-0.2 * sqrt(sum1 / d)) - exp(sum2 / d) + 20 + exp(1)


r_min, r_max = -10.0, 10.0
xaxis = arange(r_min, r_max, 0.1)
yaxis = arange(r_min, r_max, 0.1)
x, y = meshgrid(xaxis, yaxis)

res = [[0] * len(x) for i in range(len(y))]
for i in range(len(x)):
    for j in range(len(y)):
        unit = np.array([x[i][j], y[i][j]])
        res[i][j] = f(unit)
results = np.array(res)

print(results)

figure = pyplot.figure()
axis = figure.gca(projection='3d')
axis.plot_surface(x, y, results, cmap='jet')
pyplot.show()
