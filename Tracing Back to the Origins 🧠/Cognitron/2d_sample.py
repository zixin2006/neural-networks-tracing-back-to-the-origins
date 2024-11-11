import numpy as np
from cognitron import CognitronLayer

input_layer = np.random.rand(4, 4)  
cognitron_layer = CognitronLayer(input_size=(4, 4), layer_size=(2, 2), receptive_field_size=(2, 2))

output_activations = cognitron_layer.forward(input_layer)
print("Input Layer:\n", input_layer)
print("Output Activations:\n", output_activations)