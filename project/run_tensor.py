"""
Be sure you have minitorch installed in your virtual environment.
>>> pip install -Ue .
"""

from __future__ import annotations

import time
from typing import Callable, Optional, List

import minitorch


def RParam(*shape) -> minitorch.Parameter:
    """Initialize a parameter uniformly in [-1, 1].

    Args:
        *shape: Tensor shape for the parameter value.

    Returns:
        A :class:`minitorch.Parameter` holding a randomly-initialized Tensor.
    """
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    """A simple MLP: 2 -> H -> H -> 1 with ReLU and final Sigmoid."""

    def __init__(self, hidden_layers: int) -> None:
        """Create three linear layers.

        Args:
            hidden_layers: Number of neurons in the hidden layers.
        """
        super().__init__()
        # 2 -> H -> H -> 1
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        """Forward pass.

        Converts input to a Tensor and normalizes shapes:
        - single example -> (1, 2)
        - batch -> (N, 2)

        Then applies two ReLUs and a final Sigmoid.

        Args:
            x: A point or a batch of points (list/tuple/tensor).

        Returns:
            Tensor of probabilities with shape (N, 1) or (1, 1).
        """
        # Ensure the input is a Tensor
        if not isinstance(x, minitorch.Tensor):
            try:
                x = minitorch.tensor(x)
            except Exception:
                x = minitorch.tensor([x])

        # Normalize single example to shape (1, 2)
        if x.dims == 1:
            x = x.contiguous().view(1, x.shape[0])

        h1 = self.layer1(x).relu()
        h2 = self.layer2(h1).relu()
        out = self.layer3(h2)
        return out.sigmoid()  # (N, 1) or (1, 1)


class Linear(minitorch.Module):
    """Linear layer y = x @ W + b with W: (in, out), b: (out,)"""

    def __init__(self, in_size: int, out_size: int) -> None:
        """Initialize weights and bias.

        Args:
            in_size: Size of the input feature vector.
            out_size: Size of the layer output.
        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weights = RParam(in_size, out_size)  # (in, out)
        self.bias = RParam(out_size)              # (out,)

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        """Forward pass of the linear layer.

        Converts input to shape (N, in), then performs x @ W + b.

        Args:
            x: Input tensor of shape (N, in) or (in,).

        Returns:
            Tensor of shape (N, out).
        """
        if not isinstance(x, minitorch.Tensor):
            x = minitorch.tensor(x)

        # Ensure shape is (N, in)
        if x.dims == 1:
            x = x.contiguous().view(1, x.shape[0])
        elif x.dims != 2:
            raise ValueError(f"Linear expects 1D or 2D input, got {x.shape}")

        N = int(x.shape[0])
        in_size = int(x.shape[1])
        assert in_size == self.in_size, f"Expected in_size={self.in_size}, got {in_size}"

        W = self.weights.value  # (in, out)
        b = self.bias.value     # (out,)

        # Matrix multiply via elementwise ops to use the tensor backend:
        # (N, in, 1) * (1, in, out) -> (N, in, out) -> sum(dim=1) -> (N, out)
        x3 = x.contiguous().view(N, self.in_size, 1)
        W3 = W.contiguous().view(1, self.in_size, self.out_size)
        y = (x3 * W3).sum(dim=1).view(N, self.out_size) + b.contiguous().view(1, self.out_size)
        return y


def default_log_fn(
    epoch: int,
    total_loss: float,
    correct: int,
    losses: List[float],
    epoch_time_s: Optional[float] = None,
) -> None:
    """Default epoch logger.

    Args:
        epoch: Epoch number (1-based).
        total_loss: Sum of losses over the epoch.
        correct: Number of correctly classified samples.
        losses: History of total losses per epoch.
        epoch_time_s: Duration of the epoch in seconds (optional).
    """
    if epoch_time_s is None:
        print(f"Epoch {epoch:4d} | loss {total_loss:.6f} | correct {correct}")
    else:
        print(
            f"Epoch {epoch:4d} | loss {total_loss:.6f} | correct {correct} | "
            f"time {epoch_time_s * 1000:.2f} ms"
        )


class TensorTrain:
    """Training wrapper for an MLP using the MiniTorch tensor backend."""

    def __init__(self, hidden_layers: int) -> None:
        """Store hidden size and build the model.

        Args:
            hidden_layers: Number of neurons in the hidden layers.
        """
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x) -> minitorch.Tensor:
        """Run a single example through the model.

        Args:
            x: A tuple (x1, x2) or a compatible format.

        Returns:
            Model output (probability tensor).
        """
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X) -> minitorch.Tensor:
        """Run a batch through the model.

        Args:
            X: Batch of inputs with shape (N, 2).

        Returns:
            Model output (probability tensor).
        """
        return self.model.forward(minitorch.tensor(X))

    def train(
        self,
        data,
        learning_rate: float,
        max_epochs: int = 500,
        log_fn: Callable = default_log_fn,
    ) -> None:
        """Train the model and log per-epoch timing.

        Architecture: 2 -> H (ReLU) -> H (ReLU) -> 1 (Sigmoid)  
        Loss: binary cross-entropy expressed via log terms.

        Args:
            data: Dataset with attributes X (N,2), y (N,), and N (size).
            learning_rate: Learning rate for SGD.
            max_epochs: Number of epochs to train.
            log_fn: Logging callback that accepts (epoch, total_loss,
                correct, losses, epoch_time_s).
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)  # (N, 2)
        y = minitorch.tensor(data.y)  # (N,)

        losses: List[float] = []
        epoch_times: List[float] = []

        train_start = time.perf_counter()
        for epoch in range(1, self.max_epochs + 1):
            epoch_t0 = time.perf_counter()

            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)  # (N, 1) -> (N,)
            prob = (out * y) + (out - 1.0) * (y - 1.0)
            loss = -prob.log()

            # Mean loss and backward
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Parameter update
            optim.step()

            # Epoch timing
            epoch_time = time.perf_counter() - epoch_t0
            epoch_times.append(epoch_time)

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y_true = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y_true).sum()[0])
                log_fn(epoch, total_loss, correct, losses, epoch_time)

        total_time = time.perf_counter() - train_start
        if epoch_times:
            avg_epoch = sum(epoch_times) / len(epoch_times)
            print(
                f"\nTraining finished: {len(epoch_times)} epochs | "
                f"avg {avg_epoch * 1000:.2f} ms/epoch | total {total_time:.3f} s"
            )


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
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
