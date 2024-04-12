#!/usr/bin/env python
# Manually calculate the categorical cross-entropy loss for a single hard-coded sample.
# Here log is ln (the natural logarithm).
import math

softmax_output = [ 0.70,  0.10,  0.20] # Example output from the Softmax activation function on the output layer of the network for a single sample, representing a normalized probability distribution
target_output  = [ 1   ,  0   ,  0   ] # A 'one hot' target vector, meaning a '1' value in the desired output class, and '0' everywhere else

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)

