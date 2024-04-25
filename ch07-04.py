#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2 * x ** 2

x = np.array(np.arange(0, 5, 0.001))
y = f(x) # really?
C = ['k', 'g', 'r', 'b', 'c']

plt.plot(x, y)

def tangent_line(x, m, b):
    return m * x + b

for i in range(5):
    delta = 0.0001
    x1 = i
    x2 = x1 + delta
    y1 = f(x1)
    y2 = f(x1 + delta)
    m  = (y2 - y1) / (x2 - x1)
    b  = y2 - (m * x2)
    t  = [x1 - 0.9, x1, x1 + 0.9]
    plt.scatter(x1, y1, c=C[i])
    plt.plot([point for point in t], [tangent_line(point, m, b) for point in t], c=C[i])

plt.show()

