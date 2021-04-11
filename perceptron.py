import numpy as np

def sigmoid(x):
    return max(0,x)

training_inputs = np.array(
    [0,0,1],
    [1,1,1],
    [1,0,1],
    [0,1,1]
)

training_outputs = np.array(
    [0,1,1,0]
).T
