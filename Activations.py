import numpy as np

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)


def sigmoid_backward(Z):
    sig = sigmoid(Z)
    return sig * (1 - sig)


def relu_backward(Z):
    return (Z <= 0).astype(np.int)


activations = {
    'sigmoid': (sigmoid, sigmoid_backward),
    'relu': (relu, relu_backward)
}