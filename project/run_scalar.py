"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""
import random

import minitorch


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        """
        Initialize the network with two hidden layers and one output layer.

        Args:
            hidden_layers: Number of units in each hidden layer (H).
        """
        super().__init__()
        # Input 2 -> hidden H
        self.layer1 = Linear(2, hidden_layers)
        # Hidden H -> hidden H
        self.layer2 = Linear(hidden_layers, hidden_layers)
        # Hidden H -> output 1
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x: Tuple/list of two `minitorch.Scalar` inputs (x1, x2).

        Returns:
            A `minitorch.Scalar` representing P(y=1|x) after a sigmoid.
        """
        middle = [h.relu() for h in self.layer1.forward(x)]
        end = [h.relu() for h in self.layer2.forward(middle)]
        return self.layer3.forward(end)[0].sigmoid()


class Linear(minitorch.Module):
    """
    Fully-connected linear layer operating on `minitorch.Scalar`s.

    Computes y_j = bias_j + sum_i x_i * weight_{i,j} for j in 0..out_size-1.
    """
    def __init__(self, in_size, out_size):
        """
        Initialize weights and biases with uniform noise in [-1, 1).

        Args:
            in_size: Number of input features.
            out_size: Number of output features.
        """
        super().__init__()
        self.weights = []
        self.bias = []
        # Create Parameter matrix of shape (in_size, out_size)
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                    )
                )
        # Create bias vector of length out_size
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    def forward(self, inputs):
        """
        Compute the affine transform for a batch-less scalar list.

        Args:
            inputs: Iterable of length `in_size` with `minitorch.Scalar` values.

        Returns:
            List of length `out_size` with `minitorch.Scalar` outputs.

        Raises:
            AssertionError: If the number of inputs does not equal `in_size`.
        """
        # inputs: sequence of Scalar with length in_size
        # return: list of out_size Scalar
        in_size = len(self.weights)
        out_size = len(self.bias)
        assert len(inputs) == in_size, "Linear.forward: wrong input size"

        outputs = []
        for j in range(out_size):
            # Start from bias_j
            acc = self.bias[j].value
            # Accumulate sum_i x_i * w_{i,j}
            for i in range(in_size):
                acc = acc + inputs[i] * self.weights[i][j].value
            outputs.append(acc)
        return outputs


def default_log_fn(epoch, total_loss, correct, losses):
    """
    Default logger for training loops.

    Args:
        epoch: Current epoch number (1-based).
        total_loss: Sum of per-example loss for the epoch (float).
        correct: Number of correctly classified points in the epoch.
        losses: List of epoch losses accumulated so far.
    """
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class ScalarTrain:
    """
    Thin training harness for scalar-based MLP experiments.
    """
    def __init__(self, hidden_layers):
        """
        Construct a trainer and initialize the model.

        Args:
            hidden_layers: Number of hidden units for the network.
        """
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers)

    def run_one(self, x):
        """
        Run a single forward pass on a Python tuple of floats.

        Args:
            x: Tuple (x1, x2) of Python floats.

        Returns:
            Output `minitorch.Scalar` after sigmoid.
        """
        return self.model.forward(
            (minitorch.Scalar(x[0], name="x_1"), minitorch.Scalar(x[1], name="x_2"))
        )

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        """
        Train the network on a MiniTorch dataset using SGD.

        Loss:
            Negative log-likelihood for a Bernoulli target:
                If y == 1:  -log(sigmoid(x))
                If y == 0:  -log(1 - sigmoid(x))

        Args:
            data: A MiniTorch dataset with fields `N`, `X`, and `y`.
            learning_rate: Step size for SGD.
            max_epochs: Number of epochs to train.
            log_fn: Callable `(epoch, total_loss, correct, losses)` for logging.
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward + accumulate loss and correctness
            loss = 0
            for i in range(data.N):
                x_1, x_2 = data.X[i]
                y = data.y[i]
                x_1 = minitorch.Scalar(x_1)
                x_2 = minitorch.Scalar(x_2)
                out = self.model.forward((x_1, x_2))

                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0
                else:
                    prob = -out + 1.0
                    correct += 1 if out.data < 0.5 else 0
                loss = -prob.log()
                (loss / data.N).backward()
                total_loss += loss.data

            losses.append(total_loss)

            # Parameter update
            optim.step()

            # Periodic logging
            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, total_loss, correct, losses)


# if __name__ == "__main__":
#     PTS = 50
#     HIDDEN = 2
#     RATE = 0.5
#     data = minitorch.datasets["Simple"](PTS)
#     ScalarTrain(HIDDEN).train(data, RATE)

# if __name__ == "__main__":
#     PTS = 50
#     data = minitorch.datasets["Xor"](PTS)

#     HIDDEN = 10
#     RATE = 0.5
#     ScalarTrain(HIDDEN).train(data, RATE)


# datasets = {
#     "Simple": simple,
#     "Diag": diag,
#     "Split": split,
#     "Xor": xor,
#     "Circle": circle,
#     "Spiral": spiral,
# }


if __name__ == "__main__":
    PTS = 50
    data = minitorch.datasets["Spiral"](PTS)

    HIDDEN = 10
    RATE = 0.5
    ScalarTrain(HIDDEN).train(data, RATE)
