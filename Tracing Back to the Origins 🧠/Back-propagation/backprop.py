import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
    
    def feedforward(self, inputs):
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden)
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        
        final_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        final_output = sigmoid(final_input)
        return final_output

    def backpropagate(self, inputs, expected_output, learning_rate):
        # Feedforward to get the output
        actual_output = self.feedforward(inputs)
        
        # Calculate the output layer error
        output_error = expected_output - actual_output
        output_delta = output_error * sigmoid_derivative(actual_output)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer_output)
        
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate

