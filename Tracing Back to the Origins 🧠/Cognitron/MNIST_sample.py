# pip install tensorflow numpy if you don't have these packages

import numpy as np
from cognitron import CognitronLayer
import tensorflow as tf

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data

cognitron_layer = CognitronLayer(input_size=(28, 28), layer_size=(14, 14), receptive_field_size=(2, 2))

print("Testing the Cognitron-inspired layer on MNIST data")

# 3 images for testing, add if you wish
for idx in range(3):  
    input_image = x_train[idx]
    output_activations = cognitron_layer.forward(input_image)
    print(f"Image {idx+1} - Digit: {y_train[idx]}")
    print("Output Activations:\n", output_activations)

