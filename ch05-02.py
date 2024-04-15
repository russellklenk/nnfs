#!/usr/bin/env python
# Retrieve values from a softmax distribution corresponding to target values.
import math
import numpy as np

# Some made-up outputs from the softmax activation function.
# These represent a normalized probability distribution.
# Rows correspond to network outputs (passed through softmax) for a single training sample.
# Columns correspond to outputs for a single neuron in the output layer.
# Note that these are outputs from running softmax on the "right-most" hidden layer of the network.
softmax_output = [
    [ 0.70,  0.10,  0.20 ], # S0 [ N0  N1  N2 ]
    [ 0.10,  0.50,  0.40 ], # S1 [ N0  N1  N2 ]
    [ 0.02,  0.90,  0.08 ]  # S2 [ N0  N1  N2 ]
]

# Some made up target outputs.
# 0 => dog
# 1 => cat
# 2 => human
# This is a short-hand version of the one-hot target output vectors from ch05-01.py.
# Each item in `class_targets` specifies the index where the `1` would appear in the one-hot vector.
# Recall that in the one-hot target vector, a `1` appears in the desired class, and a `0` appears everywhere else.
# Writing it out long-form we'd have the following:
# target_outputs = [
#     [ 1, 0, 0 ], # Item at index 0 is set for sample 0
#     [ 0, 1, 0 ], # Item at index 1 is set for sample 1
#     [ 0, 1, 0 ]  # Item at index 2 is set for sample 2
# ]
class_targets = [0, 1, 1]

# Previously, we calculated the categorical cross-entropy loss as follows (for a single row of `softmax_output` and a single row of `target_outputs`):
# loss = -(math.log(softmax_output[0]) * target_output[0] +
#          math.log(softmax_output[1]) * target_output[1] +
#          math.log(softmax_output[2]) * target_output[2])
# 
# Since the one-hot vector has a `1` in the location of the target class and `0` everywhere else, most of these terms are 0, and only a single term matters.
# So, the above simplifies to the following:
# loss = -math.log(softmax_output[sample_index][class_targets[sample_index]])

# The following bit of code extracts the softmax_output value to input to the cross-entropy calculation.
# That is, for each row (sample), this extracts the term softmax_output[sample_index][class_targets[sample_index]].
# The result can be thought of as a 3x1 column vector, with each value corresponding to a _sample_.
# 
# Using numpy, the following could be written with the odd formulation:
# targets = softmax_outputs[[0, 1, 2], class_targets]
# Where the [0, 1, 2] is telling numpy how to index the first dimension (row 0, then row 1, then row 2)
# and class_targets is telling numpy how to index the second dimension.
targets = []
for target_index, distribution in zip(class_targets, softmax_output):
    # Note: Do not use target_index to write targets here.
    # The target_index value will be 0, then 1, then 1.
    targets.append(distribution[target_index])

print(targets)

softmax_output_array = np.array(softmax_output)
targets_array = softmax_output_array[[0, 1, 2], class_targets]
print(targets_array)
print(targets_array.shape)

# Now calculating the loss values is straightforward.
# Though log(0) is not defined so that case needs to be handled.
loss_values = [-math.log(x) for x in targets]
print(loss_values)

# Calculate the average loss per-batch as the arithmetic mean:
average_loss = sum(loss_values) / len(loss_values)
print(average_loss)

# And the above in numpy:
loss_values = -np.log(targets_array)
average_loss = np.mean(loss_values)
print(average_loss)

