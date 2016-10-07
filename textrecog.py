import os
import gzip
import numpy as np
from urllib import urlretrieve
np.seterr(all = 'ignore')

class NN(object):
    def __init__(self):
        self.weight1 = np.random.uniform(-1.0, 1.0, size=(100,784))
        self.weight2 = np.random.uniform(-1.0, 1.0, size=(10,100))
        a1 = np.zeros(784)
        a2 = np.zeros(100)
        a3 = np.zeros(10)
        self.error = 0

    def feedforward(self, x_input):
        """Return the output of the network if "a" is input."""
        w1 = self.weight1
        w2 = self.weight2
        hidden_layer = np.zeros(100)
        output_layer = np.zeros(10)

        hidden_layer =  sigmoid(np.dot(w1, x_input))
        self.a2 = hidden_layer

        output_layer = sigmoid(np.dot(w2, hidden_layer))
        self.a3 = output_layer
        return output_layer

    def back_propergation(self, input_x, input_y):
        # error calculate J(theta)
        runrate = 0.05
        delta2 = np.zeros(100)
        delta3 = np.zeros(10)
        w1 = self.weight1
        w2 = self.weight2
        self.a1 = input_x

        self.feedforward(self.a1)

        self.error += np.sum(((self.a3- input_y)**2))/2

        for k in range(10):
            z3 = np.dot(w2[k], self.a2)
            a = self.a3[k] - input_y[k]
            delta3[k] = a * dsigmoid(z3)
        for j in range(100):
            for k in range(10):
                w2[k][j] -= runrate * delta3[k] * self.a2[j]
                delta2[j] += w2[k][j] * delta3[k]
        self.weight2 = w2

        for j in range(100):
            z2 = np.dot(w1[j], self.a1)
            delta2[j] = delta2[j] * dsigmoid(z2)

        for i in range(784):
            for j in range(100):
                w1[j][i] -= runrate * delta2[j] * self.a1[i]
        self.weight1 = w1


def load_mnist_set() :
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        urlretrieve(source + filename, filename)

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    return X_train, y_train, X_test, y_test

print("Loading data...")
X_train, y_train, X_test, y_test = load_mnist_set()
print("Loading complete!")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return sigmoid(y) * (1.0- sigmoid(y) )

E = np.eye(10)
input_x = np.reshape(X_train, (60000, 784))
input_test_x = np.reshape(X_test, (10000,784))
input_y = np.zeros(10)


network = NN()
for j in range(10):
    for i in range(500):
        input_y = E[y_train[i]]
        network.back_propergation(input_x[i], input_y)
    print (network.error)
    network.error = 0

for i in range(10):
    intput_y = E[y_test[i]]
    test_result = network.feedforward(input_test_x[i])
    max = -1
    index = -1
    for x in range(10):
        if max <= test_result[x]:
            max = test_result[x]
            index = x

    print ( str(y_test[i]) + " -> " + str(index))