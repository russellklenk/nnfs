#!/usr/bin/env python

# Manual calculation of a single neuron with three inputs.

# Inputs to a neuron are either actual input data (for the input layer)
# or outputs of neurons from the previous layer of the network.
inputs = [1, 2, 3]

# Each input also has an associated weight (except for the input layer).
# Weights are typically initialized randomly.
# A weight can be thought of as the slope of a linear function - 
# in y = mx + b, the weight is the 'm' part.
# Positive weight values are upward slope (`/`) and negative are downward slope (`\`).
# The weights are tunable parameters.
weights = [0.2, 0.8, -0.5]

# Each *neuron* (not each input!) also has an associated bias.
# In y = mx + b, the bias is the 'b' part and specifies where the line crosses the
# y-axis at x = 0.
# The bias is a tunable parameter.
bias = 2

# The output of the neuron is the product of the input and weights, with the bias added.
# The term 'output' seems misleading - really this is computing the input to the activation
# function for the neuron, I think.
output = (inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2]) + bias

print(output)

