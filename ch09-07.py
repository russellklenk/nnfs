#!/usr/bin/env python
# Manual calculation of the forward pass and derivatives of the backward pass 
# for a layer consisting of multiple neurons. Each neuron in layer N receives
# a vector of partial derivatives (the gradient) from each neuron in layer N+1.
# A sum needs to be performed across all inputs to a neuron (sum the columns).
# Extends ch09-06.py to also perform the backward pass with respect to weights
# and biases, and the ReLU activation function.

import numpy as np

# z is some example output from the ReLU activation function.
z = np.array([
    [  1.00,  2.00, -3.00, -4.00 ],
    [  2.00, -7.00, -1.00,  3.00 ],
    [ -1.00,  2.00,  5.00, -1.00 ]
])

# dvalue is the gradient coming into layer N from layer N+1.
dvalues = np.array([
    [  1.0,  1.0,  1.0 ], # Inputs from sample 0
    [  2.0,  2.0,  2.0 ], # Inputs from sample 1
    [  3.0,  3.0,  3.0 ]  # Inputs from sample 2
])

dvalues_relu = np.array([
    [  1.00,  2.00,  3.00,  4.00 ],
    [  5.00,  6.00,  7.00,  8.00 ],
    [  9.00, 10.00, 11.00, 12.00 ]
])

# There are three sets of weights, one for each neuron.
# There are four inputs to the network, and weights here are transposed.
weights = np.array([
    [  0.20,  0.80, -0.50,  1.00 ], # N0 [ I0 I1 I2 I3 ]
    [  0.50, -0.91,  0.26, -0.50 ], # N1 [ I0 I1 I2 I3 ]
    [ -0.26, -0.27,  0.17,  0.87 ]  # N1 [ I0 I1 I2 I3 ]
])

inputs = np.array([
    [  1.00,  2.00,  3.00,  2.50 ],
    [  2.00,  5.00, -1.00,  2.00 ],
    [ -1.50,  2.70,  3.30, -0.80 ]
])

biases = np.array([
    [  2.00,  3.00,  0.05 ]
])

# Sum weights related to the given input multiplied by the gradient related to
# the given neuron. This is the dot product of the weights array (4x3) with the
# row vector representing the gradient incoming to the neuron (3x1, dvalues).
# This is the derivative with respect to the inputs to this neuron.
# The derivative with respect to inputs equals the weights.
dinputs = np.dot(dvalues, weights)

# Sum weights related to the given input multiplied by the gradient related to
# the given neuron. In the transposed input array, each row contains data for
# an input for all of the samples. The columns of dvalues are related to the
# outputs of single neurons for all of the samples.
# The derivative with respect to the weights equals the inputs.
dweights = np.dot(inputs.T, dvalues)

# Sum biases over samples (axis=0).
dbiases = np.sum(dvalues, axis=0, keepdims=True)

# Recall that the derivative of ReLU is 1 where the input > 0, and 0 otherwise.
drelu = np.zeros_like(z)
drelu[z > 0] = 1.0 # Useful to visualize this!
drelu *= dvalues_relu # Chain rule

print(dinputs)
print(dweights)
print(dbiases)
print(drelu)

