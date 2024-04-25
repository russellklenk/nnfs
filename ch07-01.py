#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2 * x

x = np.array(range(5))
y = f(x) # really?

plt.plot(x, y)
plt.show()

