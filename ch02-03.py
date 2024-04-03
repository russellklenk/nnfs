#!/usr/bin/env python

# Manual calculation of the output (activation function input) for a single layer
# of three neurons, where the layer has four input values, structured as a loop
# instead of explicit calculations. The output here should exactly match the output
# from the ch2-02.py script.
from typing import List

NEURON_COUNT = 3
INPUT_COUNT = 4

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

def dot(input_vec: List[float], weight_vec: List[float]) -> float:
    assert len(input_vec) == INPUT_COUNT
    assert len(input_vec) == len(weight_vec), 'Input and weight vectors are expected to have the same dimension'
    result: float = 0.0
    for K_i in range(INPUT_COUNT):
        result += input_vec[K_i] * weight_vec[K_i]

    return result

# Calculate the outputs from each of the neurons according to the formula above.
layer_outputs = [0] * NEURON_COUNT # Three neurons => three outputs

for N_i in range(NEURON_COUNT):
    layer_outputs[N_i] = dot(inputs, weights[N_i]) + bias[N_i]

print(layer_outputs)

