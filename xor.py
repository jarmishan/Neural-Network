import numpy

from neuralnetwork import Dense, NeuralNetwork

X = numpy.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = numpy.reshape([[0], [1], [1], [0]], (4, 1, 1))


model = NeuralNetwork(
    Dense(2, 3, "tanh"),
    Dense(3, 1, "tanh")
)

model.train(X, Y, 1000, display=True)

model.evaluate(X, Y)