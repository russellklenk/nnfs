#!/usr/bin/env python

# Manual calculation of the output (activation function input) for a two layers
# of three neurons, where the first "hidden" layer has four input values and the
# second has three input values (the outputs from the first layer). The input
# values to the first layer are provided as a batch. The difference from ch03-01
# is that this script does not use numpy. A matmul and transpose operation were
# added, relative to ch02-03.
from typing import List

import numpy as np

NEURON_COUNT = 3
INPUT_COUNT = 4
BATCH_SIZE = 3

# Inputs to a neuron are either actual input data (for the input layer)
# or outputs of neurons from the previous layer of the network.
# Here a batch (size = 3) is provided, which helps to avoid fitting single
# samples at a time. The shape of the inputs went from (4,) to (3, 4).
inputs = [
    [ 1.00,  2.00,  3.00,  2.50], # Sample 0
    [ 2.00,  5.00, -1.00,  2.00], # Sample 1
    [-1.50,  2.70,  3.30, -0.80]  # Sample 2
]

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
weights_layer0 = [
    [ 0.20,  0.80, -0.50,  1.00], # N_0 => one weight per-input k_i for neuron 0
    [ 0.50, -0.91,  0.26, -0.50], # N_1 => one weight per-input k_i for neuron 1
    [-0.26, -0.27,  0.17,  0.87]  # N_2 => one weight per-input k_i for neuron 2
]

# Weights for the hidden layer 1.
# The hidden layer 1 takes as its inputs the outputs from hidden layer 0.
# There were three neurons in hidden layer 0, so the number of values in each row here is 3.
# The hidden layer 1 has three neurons, so the number of rows here is 3.
weights_layer1 = [
    [ 0.10, -0.14,  0.50], # N_0 => one weight per-input k_i for neuron 0
    [-0.50,  0.12, -0.33], # N_1 => one weight per-input k_i for neuron 1
    [-0.44,  0.73, -0.13]  # N_2 => one weight per-input k_i for neuron 2
]

# Biases for the hidden layer 0.
# Each *neuron* (not each input!) also has an associated bias.
# In y = mx + b, the bias is the 'b' part and specifies where the line crosses the
# y-axis at x = 0.
# The bias is a tunable parameter.
# Since there are three neurons, there are three bias values.
# The shape of the bias vector is (3,).
bias_layer0 = [ 2.00,  3.00,  0.50]

# Biases for the hidden layer 1.
# This layer has three neurons, and so it has three bias values.
bias_layer1 = [-1.00,  2.00, -0.50]

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


def getcol(mat: List[List[float]], colidx: int) -> List[float]:
    nrows: int = len(mat)
    ncols: int = len(mat[0])
    result: List[float] = [0.0] * nrows
    for rowidx in range(nrows):
        result[rowidx] = mat[rowidx][colidx]

    return result


def transpose(mat: List[List[float]]) -> List[List[float]]:
    srcrows: int = len(mat)
    srccols: int = len(mat[0])
    result: List[List[float]] = newmat(srccols, srcrows) # Note shape transpose here: (srcrows, srccols) => (srccols, srcrows)

    for i in range(srcrows):
        for j in range(srccols):
            result[j][i] = mat[i][j]

    return result


def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    arows: int = len(a)
    acols: int = len(a[0])
    brows: int = len(b)
    bcols: int = len(b[0])
    assert acols == brows, f'Cannot multiply matrices with different inner dimension a: ({arows}, {acols}) b: ({brows}, {bcols})'
    result: List[List[float]] = newmat(arows, bcols)

    # The matrix product is the dot product of the rows of a with the columns of b.
    for rr in range(arows):
        arow = getrow(a, rr)
        for rc in range(bcols):
            bcol = getcol(b, rc)
            result[rr][rc] = dot(arow, bcol)

    return result


def rowadd(a: List[List[float]], b: List[float]) -> List[List[float]]:
    # Given a row vector [R00, R01, R02] compute [R00 + B0, R01 + B1, R02 + B2]
    arows: int = len(a)
    acols: int = len(a[0])
    for rr in range(arows):
        for rc in range(acols):
            bias: float = b[rc]
            a[rr][rc] += bias

    return a


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
layer0_outputs = layer(inputs        , weights_layer0, bias_layer0) # Input batch => hidden layer 0; outputs are (3, 3) [#samples, #neurons]
layer1_outputs = layer(layer0_outputs, weights_layer1, bias_layer1) # hidden layer 0 outputs => hidden layer 1; outputs are (3, 3) [#samples, #neurons]

# The layer_outputs is an array with shape (3, 3) and dtype = float64.
print(np.array(layer1_outputs))

