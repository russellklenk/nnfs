#!/usr/bin/env python

# Create a DenseLayer type wrapping up the data associated with a single fully-
# connected layer of a neural network. Ensure it produces the same output as 
# prior samples for this chapter (ch03-01 and ch03-02).
from typing import List

import numpy as np


def init_datagen():
    np.random.seed(0)


class DenseLayer:
    def __init__(self, input_count: int, neuron_count: int) -> None:
        # The np.random.randn function returns a gaussian distribution with a mean of 0
        # and variance of 1 (so numbers in the range -1, +1).
        # The randomly-initialized weights are scaled to reduce their magnitude in order
        # to reduce training time (why/how???)
        self.weights: np.ndarray = np.random.randn(input_count, neuron_count).astype('float32') * 0.01
        self.outputs: np.ndarray = np.zeros((input_count, neuron_count), dtype='float32')
        self.biases : np.ndarray = np.zeros((1, neuron_count), dtype='float32')


    def forward(self, inputs: np.array) -> np.ndarray:
        # Curious - there's no transpose here, while in the prior scripts a transpose of
        # the weights array was needed. Why? In ch03-03.py, shapes did not match without
        # the transpose.
        self.outputs = np.dot(inputs.astype('float64'), self.weights.astype('float64')) + self.biases.astype('float64')
        self.outputs = self.outputs.astype('float32')
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

layer0 = DenseLayer(2, 3)
layer0.forward(X)

print(layer0.outputs[:5])

