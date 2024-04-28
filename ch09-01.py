#!/usr/bin/env python
# Manual calculation of the forward pass for a single neuron.

inputs  = [  1.0, -2.0,  3.0 ]
weights = [ -3.0, -1.0,  2.0 ]
bias    = 1.0

# Forward pass
xw0 = inputs[0] * weights[0]
xw1 = inputs[1] * weights[1]
xw2 = inputs[2] * weights[2]
z   =(xw0 + xw1 + xw2) + bias
y   = max(z, 0.0) # ReLU

print(xw0, xw1, xw2, bias)
print(z)
print(y)

# The function we're calculating looks like:
# y = ReLU(i0w0 + i1w1 + i2w2 + bias) or
# y = ReLU(sum(mul(i0, w0), mul(i1, w1), mul(i2, w2), bias))

