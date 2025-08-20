from typing import Callable
import numpy as np
from numpy._typing import NDArray
from numpy import float64
import scipy
from dataclasses import dataclass
from numba import njit


@dataclass
class NeuralNetwork:
    input_nodes: int
    hidden_nodes: int
    output_nodes: int
    learning_rate: float
    # в качестве функции активации используем сигмоиду
    activation_function: Callable = lambda x: scipy.special.expit(x)

    def __post_init__(self) -> None:
        self.wih = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

    def train_batch(self, inputs_batch: NDArray[float64], targets_batch: NDArray[float64]) -> None:
        """Train on a batch of samples for better performance"""
        self.wih, self.who = train_batch(inputs_batch, targets_batch, self.wih, self.who, self.learning_rate)

    def query(self, inputs: NDArray[float64]) -> NDArray[float64]:
        return query_single(inputs, self.wih, self.who)


@njit
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

@njit
def train_batch(inputs_batch: NDArray[float64], targets_batch: NDArray[float64], 
                wih: NDArray[float64], who: NDArray[float64], learning_rate: float):
    """
    Train on a batch of samples - much more efficient
    inputs_batch: shape (batch_size, input_nodes)
    targets_batch: shape (batch_size, output_nodes)
    """
    batch_size = inputs_batch.shape[0]
    
    # Forward pass for entire batch
    hidden_inputs = np.dot(inputs_batch, wih.T)  # (batch_size, hidden_nodes)
    hidden_outputs = sigmoid(hidden_inputs)
    final_inputs = np.dot(hidden_outputs, who.T)  # (batch_size, output_nodes)
    final_outputs = sigmoid(final_inputs)

    # Backward pass
    output_errors = targets_batch - final_outputs  # (batch_size, output_nodes)
    hidden_errors = np.dot(output_errors, who)  # (batch_size, hidden_nodes)

    # Calculate weight updates (matrix operations for entire batch)
    who_update = learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)).T, hidden_outputs) / batch_size
    wih_update = learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)).T, inputs_batch) / batch_size

    # Update weights
    who += who_update
    wih += wih_update
    
    return wih, who

@njit
def query_single(inputs, wih, who):
    inputs = inputs.reshape(1, -1)  # row vector for batch compatibility
    hidden_inputs = np.dot(inputs, wih.T)
    hidden_outputs = sigmoid(hidden_inputs)
    final_inputs = np.dot(hidden_outputs, who.T)
    final_outputs = sigmoid(final_inputs)
    return final_outputs[0]  # return first (and only) result
