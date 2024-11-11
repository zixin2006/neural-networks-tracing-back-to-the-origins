import numpy as np
from backprop import NeuralNetwork

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR inputs and expected output
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR inputs
expected_output = np.array([[0], [1], [1], [0]])      # XOR outputs

# sample network
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# training parameters
epochs = 10000
learning_rate = 0.1

# training
for epoch in range(epochs):
    nn.backpropagate(inputs, expected_output, learning_rate)

print("Testing the neural network on XOR inputs after training:")
for i in range(len(inputs)):
    predicted_output = nn.feedforward(inputs[i])
    print(f"Input: {inputs[i]} -> Predicted Output: {predicted_output.round()} (Raw Output: {predicted_output})")