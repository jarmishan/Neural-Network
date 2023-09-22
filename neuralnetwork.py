import numpy as _numpy

#activation functions for linear separability
activation_functions = {
    "tanh": (lambda x: _numpy.tanh(x), lambda x: 1 - x ** 2),
    "sigmoid": (lambda x: 1 / (1 + _numpy.exp(-x)), lambda x: x * (1 - x)),
    "relu": (lambda x: _numpy.maximum(x, 0), lambda x: _numpy.where(x>0, 1, 0))
}

class Dense:
    """A class to represent a fully connected layer in a neural network."""

    def __init__(self, input_size: int, output_size: int, activation: str) -> None:
        """
        Initialize a layer of the neural network.

        Args:
            input_size (int): The number of input neurons.
            output_size (int): The number of output neurons.
            activation (str): The name of a linearly separable function.
        """

        self.weights = _numpy.random.randn(output_size, input_size)
        self.biases = _numpy.random.randn(output_size, 1)

        self.activation, self.activation_derivative = activation_functions[activation.lower()]


    def forward_pass(self, inputs: _numpy.ndarray) -> None:
        """
        Preforms forward propgation.

        Args:
            inputs (numpy.ndarray): the input to this layer.

        Returns
            self.output(numpy.ndarray): the dot product of the inputs matrix and the weights matrix added to the biases vector.
        """

        self.inputs = inputs
        self.outputs = self.activation((_numpy.dot(self.weights, self.inputs) + self.biases))
        return self.outputs


    def backward_pass(self, output_gradient: _numpy.ndarray, learning_rate: float) -> None:
        """
        Perform backward propagation.

        Args:
            output_gradient (numpy.ndarray): The rate of change of the output with respect to its input.
            learning_rate (float): The amount the model should change in response to the error.

        Returns:
            numpy.ndarray: The gradient of the error with respect to the input
        """

        output_gradient *= self.activation_derivative(self.outputs)
        self.weights -= learning_rate * _numpy.dot(output_gradient, self.inputs.T)
        self.biases -= learning_rate * output_gradient

        return _numpy.dot(self.weights.T, output_gradient)



class NeuralNetwork:
    """A class to represent a sequential neural network."""

    def __init__(self, *layers) -> None:
        """
        Initializes the neural network with the specified layers.

        Args:
            *layers: A list of layer instances forming the neural network.
        """

        self.network = layers

    def _mean_squared_error_derivative(self, correct, prediction) -> _numpy.ndarray:
        """
        Computes the derivative of mean squared error.

        Args:
            correct (numpy.ndarray): The correct output.
            prediction(numpy.ndarray): The predicted output.

        Returns:
            float : The derivative of the mean squared error.
        """

        return 2 * _numpy.mean(prediction - correct)

    def forward_propagation(self, inputs) -> _numpy.ndarray:
        """
        Performs forward propagation through the network.

        Args:
            inputs: The input data.

        Returns:
            numpy.ndarray: The output after forward propagation.
        """

        output = inputs

        for layer in self.network:
            output = layer.forward_pass(output)

        return output

    def _back_propagate(self, output, answer, learning_rate) -> _numpy.ndarray:
        """
        Backpropagates the error through the network.

        Args:
            output (numpy.ndarray): The output of the network.
            answer (numpy.ndarray): The expected output.
            learning_rate (float): The amount the model should change in response to the error.. Defaults to 0.1.
        """

        gradient = self._mean_squared_error_derivative(answer, output)

        for layer in reversed(self.network):
            gradient = layer.backward_pass(gradient, learning_rate)


    def train(self, training_inputs, training_anwsers, epochs, *, learning_rate=0.1, display=False) -> None:
        """
        Trains the neural network using the specified inputs and answers.

        Args:
            training_inputs  (numpy.ndarray): The input data for training.
            training_answers  (numpy.ndarray): The expected output data for training.
            epochs (int): The number of training epochs.
            learning_rate (float, optional): The amount the model should change in response to the error.. Defaults to 0.1.
            display (bool, optional): Whether to display progress. Defaults to False.
        """

        for epoch in range(epochs):

            for inputs, answer in zip(training_inputs, training_anwsers):

                output = self.forward_propagation(inputs)
                self._back_propagate(output, answer, learning_rate)

            if display:
                print(f"{epoch + 1} / {epochs}")


    def evaluate(self, testing_inputs, testing_anwsers):
        """
        Evaluates the neural network using the specified inputs and answers and displays the percentage accuracy.

        Args:
            testing_inputs  (numpy.ndarray): The input data for testing.
            testing_anwsers  (numpy.ndarray): The expected output data for testing.
        """
        correct = 0

        for inputs, answer in zip(testing_inputs, testing_anwsers):
            output = self.forward_propagation(inputs)

            if round(output[0][0]) == answer[0][0]:
                correct += 1

            else:
                print(output, answer)

        print(f"accuracy: {correct / len(testing_inputs) * 100}")