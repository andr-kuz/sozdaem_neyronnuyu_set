from typing import Callable
import numpy
from numpy._typing import NDArray
from numpy import float64
import scipy


from dataclasses import dataclass

from utils import load_file, prepare_mnist_line, get_index_of_max_value_in_array

@dataclass
class NeuralNetwork:
    input_nodes: int
    hidden_nodes: int
    output_nodes: int
    learning_rate: float
    # в качестве функции активации используем сигмоиду
    activation_function: Callable = lambda x: scipy.special.expit(x)

    def __post_init__(self) -> None:
        # строим матрицы весовых коэффициентов
        # количество строк в каждой матрице определяется первым множителем, соотв. колтичеству узлов в слое, куда поступают сигналы
        # поэтому input -> hidden; hidden -> output
        # выбираем Гауссово распределение (random.normal), чтобы минимизировать количество весов с крайними значениями диапазона
        # возводим в степень -0.5 (pow)
        self.from_input_to_hidden_matrix_weights = numpy.random.normal(0., pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.from_hidden_to_output_matrix_weights = numpy.random.normal(0., pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

    # опрос нейронной сети
    def query(self, inputs: NDArray[float64]) -> NDArray[float64]:
        # преобразовать список входных значений в двумерный массив
        inputs = numpy.array(inputs, ndmin=2).T

        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.from_input_to_hidden_matrix_weights, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.from_hidden_to_output_matrix_weights, hidden_inputs)
        # рассчитать исхоящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    
    def train(self, inputs_list: NDArray[float64], targets_list: NDArray[float64]) -> None:
        # преобразовать список входных значений в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.from_input_to_hidden_matrix_weights, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входящие сигналы для выхоного слоя
        final_inputs = numpy.dot(self.from_hidden_to_output_matrix_weights, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибки выходного слоя = (целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors, распределенные пропорционально весовым
        # коэффициентам связей и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.from_hidden_to_output_matrix_weights.T, output_errors)

        # обновляем весовые коэффициенты для связей между скрытым и выходным слоями
        self.from_hidden_to_output_matrix_weights += self.learning_rate * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), numpy.transpose(hidden_outputs))

        # обновляем весовые коэффициенты для связей между входным и скрытым слоями
        self.from_input_to_hidden_matrix_weights += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))



if __name__ == '__main__':
    input_nodes = 784  # 28 weight x 28 height pixels
    hidden_nodes = 100  # arbitary
    output_nodes = 10  # we need a result in range of 0 - 9
    network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate=0.3)
    content = load_file('mnist_train_100.csv', as_lines=True)
    # training
    for record in content:
        inputs, targets = prepare_mnist_line(record)
        network.train(inputs, targets)

    # test
    content = load_file('mnist_test_10.csv', as_lines=True)
    line = content[0]
    inputs, targets = prepare_mnist_line(line)
    response = network.query(inputs)
    get_index_of_max_value_in_array(response)

