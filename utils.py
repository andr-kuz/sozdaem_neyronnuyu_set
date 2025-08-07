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

def plot_ascii(image_data:NDArray[float64], width: int=28, height: int=28):
    """Convert image data to ASCII art"""
    # ASCII characters from darkest to lightest
    ascii_chars = "@%#*+=-:. "
    
    # Reshape and normalize to 0-1 range if not already
    pixels = image_data.reshape((height, width))
    pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min())
    
    # Convert to ASCII
    for row in pixels:
        line = "".join([ascii_chars[min(int(p * len(ascii_chars)), len(ascii_chars)-1)] for p in row])
        print(line)
