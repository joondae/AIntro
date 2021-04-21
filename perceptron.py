import numpy as np

# Here we are going to do two steps
# 1 - Calculate the Output
# 2 - Compare the output to the expected Output

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Maybe put other activation functions here?


# Grid of training inputs
training_inputs = np.array([
    [1,0,1,0],
    [0,1,1,1],
    [1,1,0,1],
    [0,1,0,0],
    [0,0,0,0],
    [0,0,1,1]
])

# .T Transpose the maxtrix so it's a 1 x 6
training_outputs = np.array(
    [[0,1,1,1,0,0]]
).T

# What is this statement for?
np.random.seed(2)

# How many weights do we have?
# in1, in2, in3 each need their own weight... so 3!
# Create random weights
# Change first argument in .random() method if you want to increase the size of each subarray in training_inputs
synaptic_weights = 2 * np.random.random((4,1)) - 1

#print('Random starting synaptic weights')
#print(synaptic_weights)

for iteration in range(1):

    input_layer = training_inputs

    # Take each input and multiply it times our weights
    unsquished_outputs = np.dot(input_layer, synaptic_weights)
    # print('Sum of Weights and Inputs')
    # print(unsquished_outputs)

    # Squish our result between 0 and 1 by using our normalizing fn
    normalized_outputs = sigmoid(unsquished_outputs)
    # print('Normalized (Squished) Outputs')
    # print(normalized_outputs)
    # print('derivative of normalized outputs')
    # print(sigmoid_derivative(normalized_outputs))

    # 2 - Calc error by checking the training outputs with our calulated ones
    error = training_outputs - normalized_outputs
    # print('Error')
    # print(error)

    adjustments = error * sigmoid_derivative(normalized_outputs)
    print('Adjustments')
    print(adjustments)
    print(input_layer.T)
    print(np.dot(input_layer.T, adjustments))
    print(synaptic_weights)
    synaptic_weights += np.dot(input_layer.T, adjustments)
    # print('New Weights')

print('Final synaptic weights')
print(synaptic_weights)
