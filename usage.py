import numpy as np
import matplotlib.pyplot as plt
from DNN import NN, DenseLayer

def generated_data(op):
    X, Y, x0, x1 = [], [], [], []
    for i in range(0, 2):
        for j in range(0, 2):
            X.append([i, j])
            if op(i, j) == 0:
                Y.append([0])
                x0.append([i, j])
            else:
                Y.append([1])
                x1.append([i, j])
    return np.array(X).T, np.array(Y).T, np.array(x0), np.array(x1)


def prediction(X, prediction):
    px1 = []
    px0 = []
    for x, y in zip(X.T, prediction.T):
        if y == 1:
            px1.append(x)
        else:
            px0.append(x)
    return np.array(px0), np.array(px1)


model = NN(layers=[
    DenseLayer(1, 2, activation='sigmoid')
])
X, Y, x0, x1 = generated_data(lambda x, y: x & y)
cost = model.train(X, Y, iteration=6000)
px0, px1 = prediction(X, model.predict(X) > 0.5)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(px1[:, 0], px1[:, 1], 'b.', label='x & y = 1')
ax1.plot(px0[:, 0], px0[:, 1], 'r.', label='x & y = 0')
ax1.legend()
ax2.plot(cost[:, 0], cost[:, 1], 'g')
ax2.set_xlabel('Iteration vs Cost')
plt.show()