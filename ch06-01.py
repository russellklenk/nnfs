#!/usr/bin/env python
# Attempt at "optimization" via random weight changes.
from typing import List

import numpy as np
import matplotlib.pyplot as plt


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


def vertical_data(samples, classes):
    X = np.zeros((samples*classes, 2), dtype='float32')
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        X[ix] = np.c_[np.random.randn(samples).astype('float32')*.1 + (class_number)/3, np.random.randn(samples).astype('float32')*.1 + 0.5];
        y[ix] = class_number

    return X, y


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
X, y = vertical_data(samples=100, classes=3)
#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
#plt.show()
##X, y = spiral_data(samples=100, classes=3)

# Define the network
dense_layer0 = DenseLayer(2, 3) # 2 inputs, three neurons
relu_layer0 = ReLULayer()       # ReLU activation function for first hidden layer
dense_layer1 = DenseLayer(3, 3) # 3 inputs, three neurons
softmax_layer1 = SoftmaxLayer() # Apply softmax to output from dense layer1 to get probability distribution
crossentropy_loss = CategoricalCrossEntropyLoss()

# Define some items to store a snapshot of the 'best' network classification state
lowest_loss = 9999999
best_weights_layer0 = dense_layer0.weights.copy()
best_weights_layer1 = dense_layer1.weights.copy()
best_biases_layer0 = dense_layer0.biases.copy()
best_biases_layer1 = dense_layer1.biases.copy()

# "Train" the network, looking to minimize the loss function.
# We want to see the loss value decrease, and the accuracy increase.
iteration_count = 100000
for iteration in range(iteration_count):
    # Generate a new set of weights:
    dense_layer0.weights = 0.05 * np.random.randn(2, 3).astype('float32') # Note matches shape of dense_layer0 - 2 inputs, 3 neurons
    dense_layer1.weights = 0.05 * np.random.randn(3, 3).astype('float32') # Note matches shape of dense_layer1 - 3 inputs, 3 neurons
    dense_layer0.biases  = 0.05 * np.random.randn(1, 3).astype('float32')
    dense_layer1.biases  = 0.05 * np.random.randn(1, 3).astype('float32')

    # Perform a forward pass of training data through the network.
    dense_layer0  .forward(X)
    relu_layer0   .forward(dense_layer0.outputs)
    dense_layer1  .forward(relu_layer0 .outputs)
    softmax_layer1.forward(dense_layer1.outputs)
    loss = crossentropy_loss.calculate(softmax_layer1.outputs, y)

    # Calculate how often the predicted output class matches the expected output class.
    predictions = np.argmax(softmax_layer1.outputs, axis=1)
    accuracy = np.mean(predictions == y)

    # Checkpoint if loss has decreased.
    if loss < lowest_loss:
        print(f'Checkpoint at iteration {iteration+1}/{iteration_count}: loss {loss}, acc {accuracy}.')
        best_weights_layer0 = dense_layer0.weights.copy()
        best_weights_layer1 = dense_layer1.weights.copy()
        best_biases_layer0  = dense_layer0.biases.copy()
        best_biases_layer1  = dense_layer1.biases.copy()
        lowest_loss = loss

