#!/usr/bin/env python
# Manual calculation of the forward pass and derivatives of the backward pass 
# for a single neuron.

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

# Backward pass

# Use an input value of '1' coming from the subsequent layer.
dvalue = 1.0

drelu_dz = dvalue * (1.0 if z > 0 else 0.0)

# The partial derivative of the sum operation is always 1.
dsum_diw0   = 1.0
drelu_diw0  = drelu_dz * dsum_diw0

dsum_diw1   = 1.0
drelu_diw1  = drelu_dz * dsum_diw1

dsum_diw2   = 1.0
drelu_diw2  = drelu_dz * dsum_diw2

dsum_dbias  = 1.0
drelu_dbias = drelu_dz * dsum_dbias
print(drelu_diw0, drelu_diw1, drelu_diw2, drelu_dbias)

# The partial derivative of the mul operation is whatever the input is multiplied by. For f(x, y) = x * y:
# pdx(x, y) = y, pdy(x, y) = x
dmul_di0  = weights[0]
drelu_di0 = drelu_diw0 * dmul_di0

dmul_di1  = weights[1]
drelu_di1 = drelu_diw1 * dmul_di1

dmul_di2  = weights[2]
drelu_di2 = drelu_diw2 * dmul_di2

dmul_dw0  = inputs[0]
drelu_dw0 = drelu_diw0 * dmul_dw0

dmul_dw1  = inputs[1]
drelu_dw1 = drelu_diw1 * dmul_dw1

dmul_dw2  = inputs[2]
drelu_dw2 = drelu_diw2 * dmul_dw2
print(drelu_di0, drelu_dw0, drelu_di1, drelu_dw1, drelu_di2, drelu_dw2)

