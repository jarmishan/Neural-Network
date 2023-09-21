import numpy


sigmoid = lambda x: 1 / (1 + numpy.exp(-x))

activation_functions = {
    "tanh": (
        lambda x: numpy.tanh(x), 
        lambda x: 1 - numpy.tanh(x) ** 2
    ),
    "sigmoid": (
        sigmoid, 
        lambda x: sigmoid(x) * (1 - sigmoid(x))
    ),
    "relu": (
        lambda x: numpy.maximum(0, x),
        lambda x: numpy.where(x > 0, 1, 0)
    )
}


class Dense():
    def __init__(self, input_size, output_size, activation):
        self.activation, self.activation_derivative = activation_functions[activation.lower()]
        
        self.weights = numpy.random.randn(output_size, input_size)
        self.biases = numpy.random.randn(output_size, 1)


    def forward_pass(self, inputs):
        self.inputs = inputs
        self.outputs = (numpy.dot(self.weights, self.inputs) + self.biases)
        return self.activation(self.outputs)
        
    
    def backward_pass(self, output_gradiant, learning_rate):
        output_gradiant *= self.activation_derivative(self.outputs)
        self.weights -= learning_rate * numpy.dot(output_gradiant, self.inputs.T)
        self.biases -= learning_rate * output_gradiant

        return numpy.dot(self.weights.T, output_gradiant)
    
 
class NeuralNetwork:
    def __init__(self, *layers):
        self.network = layers

    def mean_squared_error_derivative(self, correct, prediction):
        return 2 * numpy.mean(prediction - correct)
    
    
    def forward_propogation(self, inputs):
        output = inputs
        
        for layer in self.network:
            output = layer.forward_pass(output)

        return output
    

    def back_propogate(self, output, answer, learning_rate):
        gradiant = self.mean_squared_error_derivative(answer, output)

        for layer in reversed(self.network):
            gradiant = layer.backward_pass(gradiant, learning_rate)


    def train(self, training_inputs, training_anwsers, epochs, *, learning_rate=0.1, display=False):
        for epoch in range(epochs):

            for inputs, answer in zip(training_inputs, training_anwsers):

                output = self.forward_propogation(inputs)
                self.back_propogate(output, answer, learning_rate)
            
            if display:
                print(f"{epoch + 1} / {epochs}")


    def evaluate(self, testing_inputs, testing_anwsers):
        correct = 0

        for inputs, answer in zip(testing_inputs, testing_anwsers):
            output = self.forward_propogation(inputs)

            if round(output[0][0]) == answer[0][0]:
                correct += 1

            else:
                print(output, answer)

        print(f"accuracy: {correct / len(testing_inputs) * 100}")


X = numpy.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = numpy.reshape([[0], [1], [1], [0]], (4, 1, 1))


model = NeuralNetwork(
    Dense(2, 3, "tanh"),
    Dense(3, 1, "tanh")  
)

model.train(X, Y, 1000, display=True)

model.evaluate(X, Y)
