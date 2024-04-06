#!/usr/bin/env python
# Portions copyright (c) 2015 Andrej Karpathy
# License: https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
# Sources: https://cs231n.github.io/neural-networks-case-study/

import numpy as np
import matplotlib.pyplot as plt

def init_datagen():
    np.random.seed(0)

def sine_data(samples=1000):
    X = np.arange(samples).reshape(-1, 1) / samples
    y = np.sin(2 * np.pi * X).reshape(-1, 1)
    return X, y

def spiral_data(samples, classes):
    X = np.zeros((samples*classes, 2), dtype='float32')
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples).astype('float32')*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number

    return X, y

def vertical_data(samples, classes):
    X = np.zeros((samples*classes, 2), dtype='float32')
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        X[ix] = np.c_[np.random.randn(samples).astype('float32')*.1 + (class_number)/3, np.random.randn(samples).astype('float32')*.1 + 0.5];
        y[ix] = class_number

    return X, y


init_datagen()
X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

