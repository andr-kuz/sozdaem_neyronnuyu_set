import time
import numpy
import argparse
from neural_network import NeuralNetwork
from utils import load_file, prepare_mnist_line, plot_ascii


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and test a neural network on MNIST data.')
    parser.add_argument('hidden_nodes', type=int, nargs='?', default=200, help='Number of nodes in the hidden layer (default: 200)')
    parser.add_argument('learning_rate', type=float, nargs='?', default=0.3, help='Learning rate for the neural network (default: 0.3)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs (default: 1)')
    args = parser.parse_args()

    start_time = time.time()
    input_nodes = 784  # 28 weight x 28 height pixels
    output_nodes = 10  # we need a result in range of 0 - 9
    
    print(f"Initializing network with: hidden_nodes={args.hidden_nodes}, learning_rate={args.learning_rate}")
    network = NeuralNetwork(input_nodes, args.hidden_nodes, output_nodes, learning_rate=args.learning_rate)
    
    # training
    content = load_file('mnist_train.csv', as_lines=True)
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch + 1}/{args.epochs}")
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
    
    accuracy = (score / len(content)) * 100
    print(f'Effectiveness = {accuracy:.2f}%')
    end_time = time.time()
    print(f'Execution time: {(end_time - start_time):.2f} seconds')

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
