#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def f(x): # f(x) = 2x^2
    return 2 * x ** 2

x = np.array(range(5))
y = f(x) # really?

#plt.plot(x, y)
#plt.show()

# Manually calculate an approximate derivative at f(1).
# This is the slope ('rise over run') at point y = f(x).
# Delta would be infinitely small, but cannot be due to finite precision.
# If delta is less than the numeric epsilon for f64, ad will produce a divide-by-zero.
delta = 0.0001
x1 = 1
x2 = x1 + delta
y1 = f(x1)
y2 = f(x2)

ad = (y2 - y1) / (x2 - x1)
print(ad)

