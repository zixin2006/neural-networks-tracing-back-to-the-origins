

def mcculloch_pitts_neuron(inputs, weights, threshold):
    """
    McCulloch-Pitts neuron model.
    
    Parameters:
    - inputs: List of binary inputs (0 or 1).
    - weights: List of weights corresponding to each input.
    - threshold: The threshold value for the neuron to fire.
    
    Returns:
    - 1 if the neuron fires (sum of weighted inputs >= threshold), else 0.
    """
    # Calculate the weighted sum of inputs
    weighted_sum = sum(i * w for i, w in zip(inputs, weights))
    
    # Neuron fires if weighted sum meets or exceeds threshold
    output = 1 if weighted_sum >= threshold else 0
    return output

# Example usage
inputs = [1, 0, 1]        # Binary inputs
weights = [0.5, 0.5, 0.5] # Weights
threshold = 1             # Threshold for the neuron to fire

output = mcculloch_pitts_neuron(inputs, weights, threshold)
print(f"The neuron output is: {output}")