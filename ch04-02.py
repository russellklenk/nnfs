#!/usr/bin/env python

# Implement a softmax activation function, both manually and using numpy.
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


def softmax(inputs: List[List[float]]) -> List[List[float]]:
    """
    Given the inputs of form:
    S0 [ N00  N01  N02  N03 ]
    S1 [ N00  N01  N02  N03 ]
    S2 [ N00  N01  N02  N03 ]
    That is, each row of inputs corresponds to the output of a layer of N neurons for a single sample,
    the softmax function computes the normalized probability values for the sample. The values in each
    row sum to 1.0, and typically the maximum normalized probability value corresponds to the predicted class.
    Since the exponentiation operation can produce very large values, subtract the maximum value N00..N03 from
    each row prior to exponentiating. Very small values are not a problem because lim e^x as x approaches 0 is 1
    and lim e^x as x approaches -inf is 0.
    """
    nrows: int = len(inputs)    # The number of samples
    ncols: int = len(inputs[0]) # The number of neurons
    outputs: List[List[float]] = newmat(nrows, ncols)
    for rr in range(nrows):
        row    = getrow(inputs, rr)
        rowmax = max(row)
        exp    =[2.71828182846 ** (z - rowmax) for z in row]
        sumexr = sum(exp)
        for rc in range(ncols):
            outputs[rr][rc] = exp[rc] / sumexr

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


class SoftmaxLayer:
    def __init__(self) -> None:
        self.outputs: np.ndarray = None


    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Here axis = 1 means exp/sum across the rows.
        rowmax: np.ndarray = np.max(inputs, axis=1, keepdims=True) # nrows column vector where each value is the maximum value on that row
        expval: np.ndarray = np.exp(inputs - rowmax)               # Non-normalized probabilities (nrows, ncols)
        sumexp: np.ndarray = np.sum(expval, axis=1, keepdims=True)
        self.outputs = expval / sumexp


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

dense_layer1 = DenseLayer(3, 3)
dense_layer1.forward(relu_layer0.outputs)

softmax_layer1 = SoftmaxLayer()
softmax_layer1.forward(dense_layer1.outputs)
#softmax_layer1.outputs = np.array(softmax(dense_layer1.outputs))

print(softmax_layer1.outputs[:5])

