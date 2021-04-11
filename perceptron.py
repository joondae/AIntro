import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

training_inputs = np.array([
    [0,0,1],
    [1,1,1],
    [1,0,1],
    [0,1,1]
])

# .T Transpose the matrix so its a 1 x 4
training_outputs = np.array(
    [0,1,1,0]
).T

np.random.seed(1)

# How many weights do we have?
# in1, in2, in3 each need their own weight... so 3!
# Create random weights
synaptic_weights = 2 * np.random.random((3,1)) - 1

print('Random starting synaptic weights')
print(synaptic_weights)

for iteration in range(1):

    input_layer = training_inputs

    # Take each input and multiply it times our weights
    unsquished_outputs = np.dot(input_layer, synaptic_weights)

    # Squish our result between 0 and 1 by using our normalizing fn
    normalized_outputs = sigmoid(unsquished_outputs)

print('Outputs after training')
print(normalized_outputs)
