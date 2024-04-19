#!/usr/bin/env python
# This script is the combination of ch04-02.py (activation functions) and ch05-03.py (categorical cross-entropy loss)
# with all of the manual (non-numpy) calculations stripped out. Additionally, this script also calculates the accuracy,
# which indicates how often the largest confidence value output for a sample actually matches the ground truth.
# This is everything needed to perform a forward pass through a model and calculate the loss function and accuracy.
# It's worth taking this script and re-implementing it with fully manual calculations in ch05-05.py.
from typing import List

import numpy as np


def init_datagen():
    np.random.seed(0)


class DenseLayer:
    def __init__(self, input_count: int, neuron_count: int) -> None:
        # The np.random.randn function returns a gaussian distribution with a mean of 0
        # and variance of 1 (so numbers in the range -1, +1).
        self.weights: np.ndarray = np.random.randn(input_count, neuron_count).astype('float32') * 0.01
        self.outputs: np.ndarray = np.zeros((input_count, neuron_count), dtype='float32')
        self.biases : np.ndarray = np.zeros((1, neuron_count), dtype='float32')


    def forward(self, inputs: np.array) -> np.ndarray:
        self.outputs = np.dot(inputs.astype('float64'), self.weights.astype('float64')) + self.biases.astype('float64')
        self.outputs = self.outputs.astype('float32')
        return self.outputs


class ReLULayer:
    def __init__(self) -> None:
        self.outputs: np.ndarray = None

    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.outputs = np.maximum(0, inputs)
        return self.outputs


class SoftmaxLayer:
    def __init__(self) -> None:
        self.outputs: np.ndarray = None


    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Here axis = 1 means exp/sum across the rows.
        rowmax: np.ndarray = np.max(inputs, axis=1, keepdims=True) # nrows column vector where each value is the maximum value on that row
        expval: np.ndarray = np.exp(inputs - rowmax)               # Non-normalized probabilities (nrows, ncols)
        sumexp: np.ndarray = np.sum(expval, axis=1, keepdims=True)
        self.outputs = expval / sumexp


class LossFunction:
    def calculate(self, output, truths) -> float:
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
        sample_losses: np.ndarray = self.forward(output, truths)
        average_loss: float = np.mean(sample_losses)
        return average_loss


class CategoricalCrossEntropyLoss(LossFunction):
    def forward(self, softmax_outputs, ground_truth) -> np.ndarray: # 1D array of float
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


def spiral_data(samples, classes):
    X = np.zeros((samples*classes, 2), dtype='float32')
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples).astype('float32')*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number

    return X, y


init_datagen()
X, y = spiral_data(samples=100, classes=3)

dense_layer0 = DenseLayer(2, 3)
dense_layer0.forward(X)

relu_layer0 = ReLULayer()
relu_layer0.forward(dense_layer0.outputs)

dense_layer1 = DenseLayer(3, 3)
dense_layer1.forward(relu_layer0.outputs)

softmax_layer1 = SoftmaxLayer()
softmax_layer1.forward(dense_layer1.outputs)

crossentropy_loss_layer1 = CategoricalCrossEntropyLoss()
loss_layer1 = crossentropy_loss_layer1.calculate(softmax_layer1.outputs, y) # y is ground truth in shorthand form (like class_targets in ch05-02.py).

print(softmax_layer1.outputs[:5])
print(loss_layer1)

# Calculate the accuracy - how often the predicted output class matches the expected output class.
predictions = np.argmax(softmax_layer1.outputs, axis=1)
accuracy = np.mean(predictions == y)
print(accuracy)

