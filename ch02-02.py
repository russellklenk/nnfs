#!/usr/bin/env python

# Manual calculation of the output (activation function input) for a single layer
# of three neurons, where the layer has four input values.

# Inputs to a neuron are either actual input data (for the input layer)
# or outputs of neurons from the previous layer of the network.
inputs = [1, 2, 3, 2.5]

# Each input also has an associated weight (except for the input layer).
# Weights are typically initialized randomly.
# A weight can be thought of as the slope of a linear function - 
# in y = mx + b, the weight is the 'm' part.
# Positive weight values are upward slope (`/`) and negative are downward slope (`\`).
# The weights are tunable parameters.
# Note that now, there is a set of k weights for each of N neurons, where:
# - k => The number of inputs,
# - N => The number of neurons in the layer
weights = [
    [ 0.20,  0.80, -0.50,  1.00], # N_0 => one weight per-input k_i for neuron 0
    [ 0.50, -0.91,  0.26, -0.50], # N_1 => one weight per-input k_i for neuron 1
    [-0.26, -0.27,  0.17,  0.87]  # N_2 => one weight per-input k_i for neuron 2
]

# Each *neuron* (not each input!) also has an associated bias.
# In y = mx + b, the bias is the 'b' part and specifies where the line crosses the
# y-axis at x = 0.
# The bias is a tunable parameter.
# Since there are three neurons, there are three bias values.
bias = [2, 3, 0.5]

# The output of the neuron is the product of the input and weights, with the bias added.
# The output of the layer is a list of outputs for each neuron, so there are three values.
# Generally:
# O_i = dot(inputs, weights[i]) + bias
output = [
    (inputs[0] * weights[0][0] + inputs[1] * weights[0][1] + inputs[2] * weights[0][2] + inputs[3] * weights[0][3]) + bias[0],
    (inputs[0] * weights[1][0] + inputs[1] * weights[1][1] + inputs[2] * weights[1][2] + inputs[3] * weights[1][3]) + bias[1],
    (inputs[0] * weights[2][0] + inputs[1] * weights[2][1] + inputs[2] * weights[2][2] + inputs[3] * weights[2][3]) + bias[2]
]

print(output)

