from numpy import arange, square
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


def f(x):
    for i in range(x):
        yield x ** 2


a = [i for i in f(5)]
print(a)
