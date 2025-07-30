import numpy
from numpy._typing import NDArray
from numpy import float64

def load_file(file_path: str, as_lines: bool = False) -> str | list[str]:
    with open(file_path, 'r') as f:
        if as_lines:
            return f.readlines()
        else:
            return f.read()

def prepare_mnist_line(line: str) -> tuple[NDArray[float64], NDArray[float64]]:
    inputs = get_inputs_array(line)
    targets = get_targets_array(line)
    return inputs, targets

def get_inputs_array(line: str) -> NDArray[float64]:
    return (numpy.asarray(line[2:].split(','), dtype=float) / 255.0 * 0.99) + 0.01

def get_targets_array(line: str) -> NDArray[float64]:
    nodes_number = 10
    targets = numpy.zeros(nodes_number) + 0.01
    targets[int(line[0])] = 0.99
    return targets

def get_index_of_max_value_in_array(array: NDArray[float64]) -> int:
    max_val = numpy.max(array)
    index = numpy.where(array == max_val)[0][0]
    return index
