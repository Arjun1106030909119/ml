# ex09.py - tiny neural network from-scratch example

import numpy as np

# Input data (3 examples, 2 features each)
X = np.array([[2, 9],
              [1, 5],
              [3, 6]], dtype=float)

# Targets (scores out of 100)
y = np.array([[92],
              [86],
              [89]], dtype=float)

# Scale inputs and outputs
X = X / X.max(axis=0)   # scale features column-wise
y = y / 100.0           # scale targets to [0,1]

class NeuralNetwork(object):
    def __init__(self):
        # parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        # weights (random initialization)
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def feedForward(self, X):
        # forward propagation through network
        self.z = np.dot(X, self.W1)           # input -> hidden linear
        self.z2 = self.sigmoid(self.z)        # hidden activations
        self.z3 = np.dot(self.z2, self.W2)    # hidden -> output linear
        output = self.sigmoid(self.z3)        # output activation
        return output

    def sigmoid(self, s, deriv=False):
        if deriv:
            # derivative of sigmoid: s * (1 - s), here s is sigmoid(x)
            return s * (1 - s)
        return 1 / (1 + np.exp(-s))

    def backward(self, X, y, output):
        # backward propagate through the network
        self.output_error = y - output                       # error in output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)

        # how much our hidden layer weights contributed to output error
        self.z2_error = self.output_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True)

        # update weights (gradient step, no explicit learning rate here)
        # For stability you may want to multiply these updates by a small lr like 0.1
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.output_delta)

    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)


# Instantiate and train
NN = NeuralNetwork()

for i in range(1000):  # train the NN 1000 times
    if i % 100 == 0:
        # compute current loss (mean squared error)
        loss = np.mean(np.square(y - NN.feedForward(X)))
        print("Loss:", loss)
    NN.train(X, y)

print("Input: \n", X)
print("Actual Output: \n", y)
print("Final Loss: ", np.mean(np.square(y - NN.feedForward(X))))
print("\nPredicted Output: \n", NN.feedForward(X))