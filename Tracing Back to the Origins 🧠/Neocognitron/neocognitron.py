import numpy as np

# use this code and change the testing command lines related to "CognitronLayer" to "NeocognitronLayer" so you can see the test samples of Neocognitron

class NeocognitronLayer:
    def __init__(self, input_size, layer_size, receptive_field_size):
        self.input_size = input_size
        self.layer_size = layer_size
        self.receptive_field_size = receptive_field_size
        self.S_weights = np.random.rand(layer_size[0], layer_size[1], receptive_field_size[0], receptive_field_size[1])
        self.C_pool_size = (2, 2)  # Pooling size for C-cell layer

    def s_cell_activation(self, receptive_input):
        return np.sum(receptive_input)

    def forward(self, input_layer):
        S_activations = np.zeros(self.layer_size)
        
        for i in range(self.layer_size[0]):
            for j in range(self.layer_size[1]):
                start_x = i * self.receptive_field_size[0]
                start_y = j * self.receptive_field_size[1]

                receptive_input = input_layer[start_x:start_x + self.receptive_field_size[0],
                                              start_y:start_y + self.receptive_field_size[1]]
                
                activation = self.s_cell_activation(receptive_input * self.S_weights[i, j])
                S_activations[i, j] = activation

        C_activations = self.c_cell_pooling(S_activations)
        return C_activations

    def c_cell_pooling(self, S_activations):
        pooled_activations = np.zeros((self.layer_size[0] // self.C_pool_size[0], self.layer_size[1] // self.C_pool_size[1]))

        for i in range(0, self.layer_size[0], self.C_pool_size[0]):
            for j in range(0, self.layer_size[1], self.C_pool_size[1]):
                pooled_activations[i // self.C_pool_size[0], j // self.C_pool_size[1]] = np.max(
                    S_activations[i:i + self.C_pool_size[0], j:j + self.C_pool_size[1]]
                )

        return pooled_activations
