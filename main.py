import time
import numpy
from neural_network import NeuralNetwork
from utils import load_file, prepare_mnist_line, plot_ascii


if __name__ == '__main__':
    start_time = time.time()
    input_nodes = 784  # 28 weight x 28 height pixels
    hidden_nodes = 200  # arbitary
    output_nodes = 10  # we need a result in range of 0 - 9
    network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate=0.3)
    content = load_file('mnist_train.csv', as_lines=True)
    # training
    for line in content:
        inputs, targets = prepare_mnist_line(line)
        network.train(inputs, targets)

    # testing
    content = load_file('mnist_test.csv', as_lines=True)
    score = 0
    for line in content:
        inputs, targets = prepare_mnist_line(line)
        response = numpy.argmax(network.query(inputs))
        answer = int(line[0])
        if response == answer:
            score += 1
        else:
            score += 0
    print(f'effectivenes = {(score / len(content)):.2f}')
    end_time = time.time()
    print(f'Execution time: {(end_time - start_time):.2f}')

    # while True:
    #     line_number = input('Enter `mnist_test.csv` query line number: ')
    #     line = content[int(line_number)]
    #     inputs, targets = prepare_mnist_line(line)
    #     response = numpy.argmax(network.query(inputs))
    #     answer = int(line[0])
    #
    #     plot_ascii(inputs)
    #     print(f'result: {response}, answer: {answer}')
    #     scorecard_array = numpy.asarray(scorecard)
    #     print(f'effectivenes = {scorecard_array.sum() / scorecard_array.size}')
