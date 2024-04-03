#!/usr/bin/env python

# Manual calculation of the output (activation function input) for a single layer
# of three neurons, where the layer has four input values, using numpy. The output
# here should exactly match the output from the ch2-02.py and ch2-03.py scripts.
from typing import List

import numpy as np

NEURON_COUNT = 3
INPUT_COUNT = 4

# Inputs to a neuron are either actual input data (for the input layer)
# or outputs of neurons from the previous layer of the network.
inputs = np.array([1, 2, 3, 2.5])

# Each input also has an associated weight (except for the input layer).
# Weights are typically initialized randomly.
# A weight can be thought of as the slope of a linear function - 
# in y = mx + b, the weight is the 'm' part.
# Positive weight values are upward slope (`/`) and negative are downward slope (`\`).
# The weights are tunable parameters.
# Note that now, there is a set of k weights for each of N neurons, where:
# - k => The number of inputs,
# - N => The number of neurons in the layer
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
bias = np.array([2, 3, 0.5])

# The output of the neuron is the product of the input and weights, with the bias added.
# The output of the layer is a list of outputs for each neuron, so there are three values.
# Generally:
# O_i = dot(inputs, weights[i]) + bias

# Numpy makes this calculation trivial.
# This line is the same whether there's a single neuron or N neurons in the layer.
layer_outputs = np.dot(weights, inputs) + bias

# The only difference from ch2-02 and ch2-03 is that the output here has type np.array.
# Since we want the output from ch2-02, ch2-03 and ch2-04 to be *exactly* the same,
# convert the resulting np.array back into a list.
print(list(layer_outputs))

# But wait - the output here is **not** exactly identical to the output from ch2-02 and
# ch2-03. For those scripts, the output was:
# [4.8, 1.21, 2.385]
# 
# But for this script, on my machine at least, the output is close, but not identical:
# [4.8, 1.2099999999999997, 2.385]
#
# Why?
# The dtype of inputs, weights and bias is np.float64.
# What happens if we unroll the np.dot manually?
# - We get exactly the same results as ch2-02 and ch2-03; that is, [4.8, 1.21, 2.385]
# It would be valuable to figure out where this error is coming from, and to think about
# whether it matters. I assume that these sorts of errors are present all over the place
# in typical numeric processing scripts, and probably make some kind of appreciable difference.
# Maybe it's just the order of the arguments? Above we had np.dot(weights, inputs) whereas
# below we have np.dot(inputs, weights[N_i]) [no, it isn't that].

layer_outputs = [0] * NEURON_COUNT

for N_i in range(NEURON_COUNT):
    layer_outputs[N_i] = np.dot(weights[N_i], inputs) + bias[N_i]

print(list(layer_outputs))

