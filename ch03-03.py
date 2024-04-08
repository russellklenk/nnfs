#!/usr/bin/env python

# Create a DenseLayer type wrapping up the data associated with a single fully-
# connected layer of a neural network. Ensure it produces the same output as 
# prior samples for this chapter (ch03-01 and ch03-02).
from typing import List

import numpy as np


class DenseLayer:
    def __init__(self, input_count: int, neuron_count: int) -> None:
        # The np.random.randn function returns a gaussian distribution with a mean of 0
        # and variance of 1 (so numbers in the range -1, +1).
        # The magnitude of the weights is reduced so that the model starts with non-zero
        # values small enough that they do not affect training.
        self.weights: np.ndarray = np.random.randn(input_count, neuron_count) * 0.01
        self.outputs: np.ndarray = np.zeros((input_count, neuron_count))
        self.biases : np.ndarray = np.zeros((1, neuron_count))


    def forward(self, inputs: np.array) -> np.ndarray:
        self.outputs = np.dot(inputs, self.weights.T) + self.biases
        return self.outputs

# Inputs to a neuron are either actual input data (for the input layer)
# or outputs of neurons from the previous layer of the network.
# Here a batch (size = 3) is provided, which helps to avoid fitting single
# samples at a time. The shape of the inputs went from (4,) to (3, 4).
inputs = np.array([
    [ 1.00,  2.00,  3.00,  2.50], # Sample 0
    [ 2.00,  5.00, -1.00,  2.00], # Sample 1
    [-1.50,  2.70,  3.30, -0.80]  # Sample 2
])

# Weights for the hidden layer 0.
# Each input also has an associated weight (except for the input layer).
# Weights are typically initialized randomly.
# A weight can be thought of as the slope of a linear function -
# in y = mx + b, the weight is the 'm' part.
# Positive weight values are upward slope (`/`) and negative are downward slope (`\`).
# The weights are tunable parameters.
# Note that now, there is a set of k weights for each of N neurons, where:
# - k => The number of inputs,
# - N => The number of neurons in the layer
# The shape of the weights matrix is (3, 4).
weights_layer0 = np.array([
    [ 0.20,  0.80, -0.50,  1.00], # N_0 => one weight per-input k_i for neuron 0
    [ 0.50, -0.91,  0.26, -0.50], # N_1 => one weight per-input k_i for neuron 1
    [-0.26, -0.27,  0.17,  0.87]  # N_2 => one weight per-input k_i for neuron 2
])

# Weights for the hidden layer 1.
# The hidden layer 1 takes as its inputs the outputs from hidden layer 0.
# There were three neurons in hidden layer 0, so the number of values in each row here is 3.
# The hidden layer 1 has three neurons, so the number of rows here is 3.
weights_layer1 = np.array([
    [ 0.10, -0.14,  0.50], # N_0 => one weight per-input k_i for neuron 0
    [-0.50,  0.12, -0.33], # N_1 => one weight per-input k_i for neuron 1
    [-0.44,  0.73, -0.13]  # N_2 => one weight per-input k_i for neuron 2
])

# Biases for the hidden layer 0.
# Each *neuron* (not each input!) also has an associated bias.
# In y = mx + b, the bias is the 'b' part and specifies where the line crosses the
# y-axis at x = 0.
# The bias is a tunable parameter.
# Since there are three neurons, there are three bias values.
# The shape of the bias vector is (3,).
bias_layer0 = np.array([ 2.00,  3.00,  0.50])

# Biases for the hidden layer 1.
# This layer has three neurons, and so it has three bias values.
bias_layer1 = np.array([-1.00,  2.00, -0.50])

# The output of the neuron is the product of the input and weights, with the bias added.
# The output of the layer is a list of outputs for each neuron, so there are three values.
# Generally:
# O_i = dot(inputs, weights[i]) + bias

def dot(a: List[float], b: List[float]) -> float:
    assert len(a) == len(b), 'Vectors are expected to have the same dimension'
    count : int = len(a)
    result: float = 0.0
    for K_i in range(count):
        result += a[K_i] * b[K_i]

    return result

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


# Now, the above are the sort of 'core operations' implemented in the most naive way possible.
# But if we look at what we are *actually doing* we have:
# layer_outputs = rowadd(matmul(layer_inputs, transpose(layer_weights)), layer_bias)
# The matmul is performing a dot product of the rows of layer_inputs with the columns of transpose(layer_weights).
# The result of transpose(layer_weights) has shape (wcols, wrows).
# The result of matmul has shape (irows, wcols).
# However, due to the transpose, this should be equivalent to performing dot products between rows of layer_inputs and rows of layer_weights (and adding the bias to the result).
# - Need to take care to ensure that the output has the correct shape.
# - Shape is defined as (nrows, ncols)
# - Shape transpose would be (ncols, nrows)
# Also important to note is that the bias applies to the _neuron_, so the operations to perform look like this:
# result[0][0] = dot(getrow(inputs, 0), getrow(weights, 0)) + biases[0]
# result[0][1] = dot(getrow(inputs, 0), getrow(weights, 1)) + biases[1]
# result[0][2] = dot(getrow(inputs, 0), getrow(weights, 2)) + biases[2]
# result[1][0] = dot(getrow(inputs, 1), getrow(weights, 0)) + biases[0]
# etc.
def layer(inputs: List[List[float]], weights: List[List[float]], biases: List[float]) -> List[List[float]]:
    inputsr: int = len(inputs)
    inputsc: int = len(inputs[0])
    weightr: int = len(weights)
    weightc: int = len(weights[0])
    assert inputsc == weightc, f'Cannot multiply matrices with different inner dimension a: ({inputsr}, {inputsc}) b: ({weightc}, {weightr})'
    result: List[List[float]] = newmat(inputsr, weightr)

    for rr in range(inputsr):
        arow = getrow(inputs, rr)
        for rc in range(weightr):
            bcol = getrow(weights, rc) # Note: Transpose is inline
            bias = biases[rc]
            result[rr][rc] = dot(arow, bcol) + bias

    return result

# Calculate the outputs from each of the neurons according to the formula above.
# layer_outputs = [0] * NEURON_COUNT # Three neurons => three outputs

#for N_i in range(NEURON_COUNT):
#    layer_outputs[N_i] = dot(inputs, weights[N_i]) + bias[N_i]

# Note that compared to the prior example, there are three differences:
# 1. Switched from multiplying a matrix (weights) and a vector (inputs) to multiplying two matrices.
# 2. The order of the arguments to np.dot is changed from (weights, inputs) to (inputs, weights).
# 3. Because of the shape of inputs (3, 4) and weights (3, 4), the weights matrix must be transposed.
# The result is now a 3x3 matrix, where each row represents the layer outputs for each sample of the batch.
# With order (inputs, weights.T) => row R_i is the layer outputs for sample S_i, for example:
#   S_0 [N_0, N_1, N_2]
#   S_1 [N_0, N_1, N_2]
#   S_3 [N_0, N_1, N_2]
# With order (weights, inputs.T) => row R_i is the neuron output, for example:
#   N_0 [S_0, S_1, S_2]
#   N_1 [S_0, S_1, S_2]
#   N_2 [S_0, S_1, S_2]
# The former order is desirable - the input to this layer was sample-oriented; sample-oriented output is fed to the next layer.
# layer0_outputs = layer(inputs        , weights_layer0, bias_layer0) # Input batch => hidden layer 0; outputs are (3, 3) [#samples, #neurons]
# layer1_outputs = layer(layer0_outputs, weights_layer1, bias_layer1) # hidden layer 0 outputs => hidden layer 1; outputs are (3, 3) [#samples, #neurons]

# The layer_outputs is an array with shape (3, 3) and dtype = float64.
# print(np.array(layer1_outputs))

layer0 = DenseLayer(4, 3)
layer0.weights = weights_layer0
layer0.biases = bias_layer0

layer1 = DenseLayer(3, 3)
layer1.weights = weights_layer1
layer1.biases = bias_layer1

layer0_outputs = layer0.forward(inputs)
layer1_outputs = layer1.forward(layer0_outputs)

print(layer1_outputs)

