#!/usr/bin/env python

# Implement a ReLU (Rectified Linear Unit) activation function, both manually
# and using numpy. ReLU is defined as f(x) = x where x > 0, f(x) = 0 where x <= 0.
from typing import List

import numpy as np


def init_datagen():
    np.random.seed(0)

# There's a design decision here - is a matrix represented as a 2D array (list of lists)?
# Or should it be represented as a 1D array with explicit shape?

def newmat(nrows: int, ncols: int) -> List[List[float]]:
    result: List[List[float]] = []
    for rr in range(nrows):
        row: List[float] = [0.0] * ncols
        result.append(row)

    return result


def getrow(mat: List[List[float]], rowidx: int) -> List[float]:
    return mat[rowidx]


def relu(inputs: List[List[float]]) -> List[List[float]]:
    nrows: int = len(inputs)
    ncols: int = len(inputs[0])
    outputs: List[List[float]] = newmat(nrows, ncols)
    for rr in range(nrows):
        for rc in range(ncols):
            outputs[rr][rc] = max(0, inputs[rr][rc])

    return outputs


class DenseLayer:
    def __init__(self, input_count: int, neuron_count: int) -> None:
        # The np.random.randn function returns a gaussian distribution with a mean of 0
        # and variance of 1 (so numbers in the range -1, +1).
        self.weights: np.ndarray = np.random.randn(input_count, neuron_count).astype('float32') * 0.01
        self.outputs: np.ndarray = np.zeros((input_count, neuron_count), dtype='float32')
        self.biases : np.ndarray = np.zeros((1, neuron_count), dtype='float32')


    def forward(self, inputs: np.array) -> np.ndarray:
        self.outputs = np.dot(inputs.astype('float64'), self.weights.astype('float64')) + self.biases.astype('float64')
        self.outputs = self.outputs.astype('float32')
        return self.outputs


class ReLULayer:
    def __init__(self) -> None:
        self.outputs: np.ndarray = None

    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.outputs = np.maximum(0, inputs)
        return self.outputs


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


init_datagen()
X, y = spiral_data(samples=100, classes=3)

dense_layer0 = DenseLayer(2, 3)
dense_layer0.forward(X)

relu_layer0 = ReLULayer()
relu_layer0.forward(dense_layer0.outputs)

print(relu_layer0.outputs[:5])

