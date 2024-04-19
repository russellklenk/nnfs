#!/usr/bin/env python
# Wrap up the code from ch05-02.py to calculate categorical cross-entropy loss into a class.
# The class handles either a list of class targets OR an array of one-hot vectors.
# TBD which representation is more useful.
import math
import numpy as np
from   typing import List

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

# The equivalent of np.clip, used to clamp all values in an input array into a given range.
def clip(items: List[float], lower: float, upper: float) -> List[float]:
    count : int = len(items)
    result: List[float] = [0.0] * count
    for index, value in enumerate(items):
        if value < lower:
            result[index] = lower
        elif value > upper:
            result[index] = upper
        else:
            result[index] = value

    return result

# The softmax output value for a given sample might be zero if the model has 100% 
# confidence in a value other than the target class, which means that the confidence
# value selected and placed in the target array will be 0.
# The value of log(0) is negative infinity, which will result in a divide-by-zero
# error when calculating the loss values and cause the average loss to become infinity.
# Clamp all of the values in the set of softmax outputs for class targets into a range
# where this cannot happen.
targets_clipped = clip(targets, 1e-7, 1.0 - 1e-7)
targets_clipped_array = np.clip(targets_array, 1e-7, 1.0 - 1e-7)

# Now calculating the loss values is straightforward.
loss_values = [-math.log(x) for x in targets_clipped]
print(loss_values)

# Calculate the average loss per-batch as the arithmetic mean:
average_loss = sum(loss_values) / len(loss_values)
print(average_loss)

# And the above in numpy:
loss_values = -np.log(targets_clipped_array)
average_loss = np.mean(loss_values)
print(average_loss)


### Class version

class LossFunction:
    def calculate(self, output, truths):
        """
        Calculate the average loss for a batch of samples.

        Parameters
        ----------
          - `output`: The set of outputs from the activation function for each sample in the batch.
          - `truths`: The ground truth for each sample in the batch. May be either shorthand form, where each element
                      specifies the index of the desired class, or longform where each element is a one-hot vector.

        Returns
        -------
          The average loss for the batch.
        """
        sample_losses = self.forward(output, truths)
        average_loss = np.mean(sample_losses)
        return average_loss


class CategoricalCrossEntropyLoss(LossFunction):
    def forward(self, softmax_outputs, ground_truth) -> List[float]:
        sample_count: int = len(softmax_outputs)

        if len(ground_truth.shape) == 1:
            # Shorthand form; ground_truth specifies the index in each sample of the element in the one-hot vector (class_targets in ch05-02.py).
            correct_confidences = softmax_outputs[range(sample_count), ground_truth]

        elif len(ground_truth.shape) == 2:
            # Longform; ground_truth is a list of one-hot vectors like target_outputs in ch05-01.py. 
            # The ground_truth is treated as a mask.
            correct_confidences = np.sum(softmax_outputs * ground_truth, axis=1)

        else:
            raise Error('Unexpected shape for ground_truth input')

        clipped_confidences = np.clip(correct_confidences, 1e-7, 1 - 1e-7)
        loss_values = -np.log(clipped_confidences)
        return loss_values


class_targets_array = np.array(class_targets)
lossfn = CategoricalCrossEntropyLoss()
average_loss_value = lossfn.calculate(softmax_output_array, class_targets_array)
print(average_loss_value)

