#!/usr/bin/env python
# Manual calculation of the forward pass and derivatives of the backward pass 
# for a layer consisting of multiple neurons. Each neuron in layer N receives
# a vector of partial derivatives (the gradient) from each neuron in layer N+1.
# A sum needs to be performed across all inputs to a neuron (sum the columns).
# Extends ch09-07.py to also perform the forward pass and update parameters 
# based on gradients (for ReLU - for example only, you would not really do this).

import numpy as np

# dvalue is the gradient coming into layer N from layer N+1.
dvalues = np.array([
    [  1.0,  1.0,  1.0 ], # Inputs from sample 0
    [  2.0,  2.0,  2.0 ], # Inputs from sample 1
    [  3.0,  3.0,  3.0 ]  # Inputs from sample 2
])

# Each row in inputs corresponds to a single sample.
inputs = np.array([
    [  1.00,  2.00,  3.00,  2.50 ],
    [  2.00,  5.00, -1.00,  2.00 ],
    [ -1.50,  2.70,  3.30, -0.80 ]
])

# There are three sets of weights, one for each neuron.
# There are four inputs to the network, and weights here are transposed.
weights = np.array([
    [  0.20,  0.80, -0.50,  1.00 ], # N0 [ I0 I1 I2 I3 ]
    [  0.50, -0.91,  0.26, -0.50 ], # N1 [ I0 I1 I2 I3 ]
    [ -0.26, -0.27,  0.17,  0.87 ]  # N1 [ I0 I1 I2 I3 ]
])

weights_transposed = weights.T

# There's one bias for each neuron (shape (1, #neurons)).
biases = np.array([
    [  2.00,  3.00,  0.05 ]
])

# Perform the forward pass.
layer_outputs = np.dot(inputs, weights_transposed) + biases # Dense layer
relu_outputs = np.maximum(0, layer_outputs) # ReLU activation layer

# Perform the backward pass for the ReLU layer.
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0.0 # A simplification of what was in ch09-07, see page 210+211.

# Perform the backward pass for the dense layer.
dinputs = np.dot(drelu, weights)
dweights = np.dot(inputs.T, drelu)
dbiases = np.sum(drelu, axis=0, keepdims=True)

# Sample optimization
weights_transposed += -0.001 * dweights
biases += -0.001 * dbiases
print(weights_transposed)
print(biases)

