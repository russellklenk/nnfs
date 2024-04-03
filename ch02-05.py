#!/usr/bin/env python

# Manual calculation of the output (activation function input) for a single layer
# of three neurons, where the layer has four input values, using numpy, with a
# batch of inputs provided (inputs change from a vector to a matrix).
from typing import List

import numpy as np

NEURON_COUNT = 3
INPUT_COUNT = 4
BATCH_SIZE = 3

# Inputs to a neuron are either actual input data (for the input layer)
# or outputs of neurons from the previous layer of the network.
# Here a batch (size = 3) is provided, which helps to avoid fitting single
# samples at a time. The shape of the inputs went from (4,) to (3, 4).
inputs = np.array([
    [ 1.00,  2.00,  3.00,  2.50], # Sample 0
    [ 2.00,  5.00, -1.00,  2.00], # Sample 1
    [-1.50,  2.70,  3.30, -0.80]  # Sample 2
])

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
weights = np.array([
    [ 0.20,  0.80, -0.50,  1.00], # N_0 => one weight per-input k_i for neuron 0
    [ 0.50, -0.91,  0.26, -0.50], # N_1 => one weight per-input k_i for neuron 1
    [-0.26, -0.27,  0.17,  0.87]  # N_2 => one weight per-input k_i for neuron 2
])

# Each *neuron* (not each input!) also has an associated bias.
# In y = mx + b, the bias is the 'b' part and specifies where the line crosses the
# y-axis at x = 0.
# The bias is a tunable parameter.
# Since there are three neurons, there are three bias values.
# The shape of the bias vector is (3,).
bias = np.array([2, 3, 0.5])

# The output of the neuron is the product of the input and weights, with the bias added.
# The output of the layer is a list of outputs for each neuron, so there are three values.
# Generally:
# O_i = dot(inputs, weights[i]) + bias

# Numpy makes this calculation trivial.
# This line is the same whether there's a single neuron or N neurons in the layer.
# 
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
layer_outputs = np.dot(inputs, weights.T) + bias

# The layer_outputs is an array with shape (3, 3) and dtype = float64.
print(layer_outputs)

