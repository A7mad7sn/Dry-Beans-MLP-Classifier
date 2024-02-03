import numpy as np


class Multilayer:
    def __init__(self, activation='hyperbolic_tangent', learning_rate=0.0001, epochs=1000, use_bias=True,
                 hidden_layers=None):
        if hidden_layers is None:
            hidden_layers = [64 ,3 ,50, 8]
        self.hidden_layers = hidden_layers
        self.input_layer = 5
        self.output_layer = 3
        self.useBias = use_bias
        self.lr = learning_rate
        self.ep = epochs
        if activation == 'hyperbolic_tangent':
            self.activation_fn = self._tanh
            self.derivative = self._tanh_drev
        elif activation == 'sigmoid':
            self.activation_fn = self._sigmoid
            self.derivative = self._segmoid_drev

        # Initialize weights and biases for hidden layers
        self.all_weights = []
        for i in range(len(hidden_layers)):
            if i == 0:
                self.all_weights.append(np.random.randn(self.input_layer, self.hidden_layers[0]))
            if i == len(hidden_layers) - 1:
                self.all_weights.append(np.random.randn(hidden_layers[i], self.output_layer))
            else:
                self.all_weights.append(np.random.randn(self.hidden_layers[i], self.hidden_layers[i + 1]))
        self.all_bias = []
        for i in range(len(hidden_layers)):
            self.all_bias.append(np.random.randn(self.hidden_layers[i]))
            if i == len(hidden_layers) - 1:
                self.all_bias.append(np.random.randn(self.output_layer))

    def train(self, X, Y_actual):
        for _ in range(self.ep):
            all_layers_output = []
            all_layers_errors = []
            # Step 1:::Forward Propagation
            current_input = X
            for l in range(len(self.all_weights)):
                if l == 0:
                    current_input = X
                if self.useBias:
                    current_output = current_input.dot(self.all_weights[l]) + self.all_bias[l].T
                else:
                    current_output = current_input.dot(self.all_weights[l])

                current_output = self.activation_fn(current_output)

                all_layers_output.append(current_output)

                current_input = current_output

            # Step 2::: Backward Propagation
            output_error = (Y_actual - all_layers_output[-1]) * self.derivative(all_layers_output[-1])
            current_error = output_error
            for l in range(len(self.all_weights) - 1, 0, -1):
                current_error = np.dot(current_error, self.all_weights[l].T) * self.derivative(all_layers_output[l - 1])
                all_layers_errors.append(current_error)

            all_layers_errors.reverse()
            all_layers_errors.append(output_error)

            # Updating weights
            for l in range(len(self.all_weights)):
                if self.useBias:
                    self.all_bias[l] += np.sum(all_layers_errors[l]) * self.lr
                if l == 0:
                    self.all_weights[l] += np.dot(X.T, all_layers_errors[l]) * self.lr
                else:
                    self.all_weights[l] += np.dot(all_layers_output[l - 1].T, all_layers_errors[l]) * self.lr

        # After Learning
        y_predicted = self.test(X)
        return y_predicted

    def test(self, X):
        current_input = current_output = 0
        for l in range(len(self.all_weights)):
            if l == 0:
                current_input = X

            if self.useBias:
                current_output = current_input.dot(self.all_weights[l]) + self.all_bias[l].T
            else:
                current_output = current_input.dot(self.all_weights[l])

            current_output = self.activation_fn(current_output)

            current_input = current_output

        y_predicted = self.assign_class(current_output)
        return y_predicted

    def assign_class(self, output_list):
        new_output_list = []
        for node_output in output_list:
            small_list = []
            for digit in node_output:
                if digit == max(node_output):
                    small_list.append(1)
                else:
                    small_list.append(0)
            new_output_list.append(small_list)

        new_output_list = np.array(new_output_list)
        return new_output_list

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _segmoid_drev(self, x):
        sigmoid_x = self._sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def _tanh(self, x):
        return (1 - np.exp(-x)) / (1 + np.exp(-x))
    def _tanh_drev(self, x):
        tan_h = self._tanh(x)
        return 1 - tan_h ** 2
