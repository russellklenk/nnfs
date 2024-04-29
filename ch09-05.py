#!/usr/bin/env python
# Manual calculation of the forward pass and derivatives of the backward pass 
# for a layer consisting of multiple neurons. Each neuron in layer N receives
# a vector of partial derivatives (the gradient) from each neuron in layer N+1.
# A sum needs to be performed across all inputs to a neuron (sum the columns).

import numpy as np

# dvalue is the gradient coming into layer N from layer N+1.
dvalues = np.array([
    [ 1.0,  1.0,  1.0 ]
])

# There are three sets of weights, one for each neuron.
# There are four inputs to the network, and weights here are transposed.
weights = np.array([
    [  0.20,  0.80, -0.50,  1.00 ], # N0 [ I0 I1 I2 I3 ]
    [  0.50, -0.91,  0.26, -0.50 ], # N1 [ I0 I1 I2 I3 ]
    [ -0.26, -0.27,  0.17,  0.87 ]  # N1 [ I0 I1 I2 I3 ]
])

# Sum weights related to the given input multiplied by the gradient related to
# the given neuron. This is the dot product of the weights array (4x3) with the
# row vector representing the gradient incoming to the neuron (3x1, dvalues).
dinputs = np.dot(dvalues[0], weights)
print(dinputs)

