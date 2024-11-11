import numpy as np

class CognitronLayer:
    def __init__(self, input_size, layer_size, receptive_field_size):
        """
        Parameters:
        - input_size: Tuple (height, width) of the input layer.
        - layer_size: Tuple (height, width) of the neuron grid in this layer.
        - receptive_field_size: Tuple (height, width) of the receptive field for each neuron.
        """
        self.input_size = input_size
        self.layer_size = layer_size
        self.receptive_field_size = receptive_field_size
        # Initialize weights randomly from the receptive field for each neuron
        self.weights = np.random.rand(layer_size[0], layer_size[1], receptive_field_size[0], receptive_field_size[1])

    def activate_neuron(self, receptive_input):
        """
        Compute the activation of a neuron based on its receptive field input.
        Using a simple sum as an activation function here for demonstration.
        """
        return np.sum(receptive_input)

    def forward(self, input_layer):
        """
        Perform a forward pass by applying the receptive field across the input layer.
        Each neuron calculates its activation, and the 'winner' strengthens its connections.
        """
        output_activations = np.zeros(self.layer_size)

        for i in range(self.layer_size[0]):
            for j in range(self.layer_size[1]):

                start_x = i * self.receptive_field_size[0]
                start_y = j * self.receptive_field_size[1]
                receptive_input = input_layer[start_x:start_x + self.receptive_field_size[0],
                                              start_y:start_y + self.receptive_field_size[1]]

                activation = self.activate_neuron(receptive_input * self.weights[i, j])

                output_activations[i, j] = activation

        # "Winner-takes-all" mechanism: only the highest activation in the receptive field is strengthened
        max_activation_index = np.unravel_index(np.argmax(output_activations, axis=None), output_activations.shape)
        self.weights[max_activation_index] += receptive_input  # Strengthening the connection

        return output_activations
