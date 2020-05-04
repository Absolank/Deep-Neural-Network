import numpy as np
from Activations import activations

class DenseLayer:
    def __init__(self, nodes=1, input_size=1, activation='relu'):
        self.nodes = nodes
        self.input_size = input_size
        self.W = np.random.randn(nodes, input_size)
        self.b = np.zeros((nodes, 1))
        self.activation, self.derivative = activations[activation]

    def forward(self, X):
        Z = np.dot(self.W, X) + self.b
        return Z, self.activation(Z)

    def backward(self, dA, Z, A_prev):
        m = A_prev.shape[1]
        dZ = dA * self.derivative(Z)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(self.W.T, dZ)
        return dW, db, dA_prev

    def update(self, alpha_dA, alpha_db):
        self.W -= alpha_dA
        self.b -= alpha_db


class NN:
    def __init__(self, layers):
        if len(layers) < 1:
            raise Exception("Number of layers should be greater than 1!!!")
        self.layers = layers

    def predict(self, X):
        A = X
        for layer in self.layers:
            z, a = layer.forward(A)
            A = a
        return A

    @staticmethod
    def loss(_Y, Y):
        return - np.sum(Y * np.log(_Y) + (1 - Y) * np.log(1 - _Y)) / Y.shape[1]

    def forward(self, X):
        Z = [X]
        A = [X]
        i = 0
        for layer in self.layers:
            z, a = layer.forward(A[i])
            Z.append(z)
            A.append(a)
            i = i + 1
        return Z, A

    def backward(self, Z, A, Y, learning_rate):
        l = len(self.layers)
        dA = np.divide(1 - Y, 1 - A[l]) - np.divide(Y, A[l])
        for layer in reversed(self.layers):
            dW, db, dA = layer.backward(dA, Z[l], A[l - 1])
            layer.update(learning_rate * dW, learning_rate * db)
            l -= 1

    def train(self, X, Y, learning_rate=0.1, iteration=100):
        loss = []
        loss_step = max(1, iteration // 100)
        for i in range(0, iteration):
            Z, A = self.forward(X)
            self.backward(Z, A, Y, learning_rate)
            l = self.loss(A[len(A) - 1], Y)
            if i % loss_step == 0:
                loss.append([i, l])
        return np.array(loss)
