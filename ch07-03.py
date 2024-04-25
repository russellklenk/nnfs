#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2 * x ** 2

x = np.array(np.arange(0, 5, 0.001))
y = f(x) # really?

# The following are constant; we want to display the tangent at y = f(2).
delta = 0.0001
x1 = 2
x2 = x1 + delta
y1 = f(x1)
y2 = f(x1 + delta)
m  = (y2 - y1) / (x2 - x1)
b  = y2 - (m * x2)

def tangent_line(x):
    return m * x + b

# Plot the function y = f(x)
plt.plot(x, y)

# Plot the tangent at f(2)
t = [2 - 0.9, 2, 2 + 0.9]
plt.plot(t, [tangent_line(x) for x in t])
plt.show()

