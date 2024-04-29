#!/usr/bin/env python
# Manual calculation of the forward pass and derivatives of the backward pass 
# for a layer consisting of multiple neurons. Each neuron in layer N receives
# a vector of partial derivatives (the gradient) from each neuron in layer N+1.
# A sum needs to be performed across all inputs to a neuron (sum the columns).

import numpy as np

# dvalue is the gradient coming into layer N from layer N+1.
dvalues = [
    [ 1.0,  1.0,  1.0 ]
]

# There are three sets of weights, one for each neuron.
# There are four inputs to the network, and weights here are transposed.
weights = [
    [  0.20,  0.80, -0.50,  1.00 ], # N0 [ I0 I1 I2 I3 ]
    [  0.50, -0.91,  0.26, -0.50 ], # N1 [ I0 I1 I2 I3 ]
    [ -0.26, -0.27,  0.17,  0.87 ]  # N1 [ I0 I1 I2 I3 ]
]

weights_transpose = [
    [  0.20,  0.50, -0.26 ], # S0 [ I1 I2 I3 ]
    [  0.80, -0.91, -0.27 ], # S1 [ I1 I2 I3 ]
    [ -0.50,  0.26,  0.17 ], # S2 [ I1 I2 I3 ]
    [  1.00, -0.50,  0.87 ]  # S3 [ I1 I2 I3 ]
]

# Sum weights related to the given input multiplied by the gradient related to
# the given neuron. This is the dot product of the weights array (4x3) with the
# row vector representing the gradient incoming to the neuron (3x1, dvalues).
dx0 = sum([weights_transpose[0][0] * dvalues[0][0], weights_transpose[0][1] * dvalues[0][1], weights_transpose[0][2] * dvalues[0][2]])
dx1 = sum([weights_transpose[1][0] * dvalues[0][0], weights_transpose[1][1] * dvalues[0][1], weights_transpose[1][2] * dvalues[0][2]])
dx2 = sum([weights_transpose[2][0] * dvalues[0][0], weights_transpose[2][1] * dvalues[0][1], weights_transpose[2][2] * dvalues[0][2]])
dx3 = sum([weights_transpose[3][0] * dvalues[0][0], weights_transpose[3][1] * dvalues[0][1], weights_transpose[3][2] * dvalues[0][2]])

dinputs = np.array([dx0, dx1, dx2, dx3])
print(dinputs)

