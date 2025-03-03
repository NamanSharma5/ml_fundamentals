import random
from grad_engine import Value
from dataclasses import dataclass


class InitializationFunctions:
    """Encapsulates standard weight initialization strategies."""
    @staticmethod
    def zeros():
        """Returns a function that initializes weights to zero."""
        return lambda: 0

    @staticmethod
    def random_uniform(low=-1.0, high=1.0):
        """Returns a function that initializes weights randomly within a range."""
        return lambda: random.uniform(low, high)


class Neuron:
    """
    think of this as traditional neuron, n inputs, 1 output
    this will use the Value class from our grad engine
    """

    def __init__(self, nin, init_weight_fn=None, activation_fn=lambda x: x):
        """
        args:
            ni - number of inputs
        nIn weights + 1 bias
        """
        if init_weight_fn is None:
            init_weight_fn = InitializationFunctions.random_uniform()
        else:
            assert callable(init_weight_fn)
        self.weights = [Value(init_weight_fn()) for _ in range(nin)]
        self.bias = Value(init_weight_fn())
        self.activation_fn = activation_fn

    def parameters(self):
        return self.weights + [self.bias]

    def forward(self, inputs):
        """
        args:
            inputs - list of n inputs
        """
        assert (len(inputs) == len(self.weights))

        weighted_inputs = map(lambda w_i: w_i[0] * w_i[1], zip(self.weights, inputs))
        tmp = sum(weighted_inputs, self.bias)
        return self.activation_fn(tmp)

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)


class Layer:

    def __init__(self, nin, nout, init_weight_fn=None, activation_fn=lambda x: x):
        """
        order of n_out matters
        """
        self.neurons = [Neuron(nin, init_weight_fn, activation_fn) for _ in range(nout)]

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)


class MLP:

    def __init__(self, nin, layer_sizes, loss_fn=None):
        """
        args:
            layer_sizes - list of integers, where each integer is the number of output neurons in that layer
        """
        self.layers = []
        n_prev = nin
        for n in layer_sizes:
            self.layers.append(Layer(n_prev, n))
            n_prev = n

    def parameters(self, verbose=False):
        if verbose:
            for layer in self.layers:
                layer: Layer
                print(layer.parameters())
        return [param for layer in self.layers for param in layer.parameters()]

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            layer: Layer
            x = layer.forward(x)
        if len(x) == 1:
            return x[0]
        return x


if __name__ == "__main__":

    @dataclass
    class TrainingConfig:
        # hyperparameters
        learning_rate = 0.001
        no_of_epochs = 20

    mlp = MLP(3, [4, 4, 1])  # 3 inputs, 2 hidden layers with 4 neurons each, and 1 output neuron
    print(f"{len(mlp.parameters())=}")

    inputs = [Value(1.0), Value(2.0), Value(3.0)]
    example_inputs = [
        [Value(1.0), Value(2.0), Value(3.0)],
        [Value(2.0), Value(3.0), Value(4.0)],
        [Value(3.0), Value(4.0), Value(5.0)]
    ]
    example_targets = [
        [Value(5.0)],
        [Value(6.0)],
        [Value(7.0)]
    ]

    for epoch in range(no_of_epochs):
        for x, y_hat in zip(example_inputs, example_targets):
            y = mlp.forward(x)
            loss: Value = (y - y_hat[0]) ** 2
            loss.backward()

        print(f"Epoch {epoch}: {loss.data=}")
        for param in mlp.parameters():
            param: Value
            # 1 step of gradient descent
            param.data -= learning_rate * param.grad

        # NOTE: Zero the gradients after updating
        for param in mlp.parameters():
            param.grad = 0.0
