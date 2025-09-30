from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    "Multiplication function"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """
        Compute a * b.

        Saves a and b for the backward pass.

        Args:
            ctx: Context used to save inputs.
            a: Left operand.
            b: Right operand.

        Returns:
            a * b.
        """
        ctx.save_for_backward(a, b)
        return float(operators.mul(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """
        Backprop for multiplication.

        Args:
            ctx: Context containing saved (a, b).
            d_output: Upstream gradient.

        Returns:
            (d_output * b, d_output * a).
        """
        a, b = ctx.saved_values
        # d/d(a) (a*b) = b,  d/d(b) (a*b) = a
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    "Inverse function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Compute 1 / a.

        Saves a for the backward pass.

        Args:
            ctx: Context used to save a.
            a: Input value.

        Returns:
            1.0 / a.
        """
        ctx.save_for_backward(a)
        return float(operators.inv(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Backprop for reciprocal.

        Args:
            ctx: Context containing saved input a.
            d_output: Upstream gradient.

        Returns:
            d_output * (-1 / a^2).
        """
        (a,) = ctx.saved_values
        # d/d(a) (1/a) = -1/a^2 -> inv_back(a, d_output)
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    "Negation function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Compute -a.

        Args:
            ctx: Context (unused).
            a: Input value.

        Returns:
            -a.
        """
        return float(operators.neg(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Backprop for negation.

        Args:
            ctx: Context (unused).
            d_output: Upstream gradient.

        Returns:
            -d_output.
        """
        # d/d(a) (-a) = -1
        return -d_output


class Sigmoid(ScalarFunction):
    "Sigmoid function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Compute sigmoid(a).

        Saves the output s for an efficient backward pass.

        Args:
            ctx: Context used to save s.
            a: Input value.

        Returns:
            sigmoid(a).
        """
        s = operators.sigmoid(a)
        ctx.save_for_backward(s)
        return float(s)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Backprop for sigmoid.

        Args:
            ctx: Context containing saved output s = sigmoid(a).
            d_output: Upstream gradient.

        Returns:
            d_output * s * (1 - s).
        """
        (s,) = ctx.saved_values
        # d/d(a) σ(a) = σ(a) * (1 - σ(a))
        return d_output * s * (1.0 - s)


class ReLU(ScalarFunction):
    "ReLU function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Compute ReLU(a).

        Saves a for the backward pass.

        Args:
            ctx: Context used to save a.
            a: Input value.

        Returns:
            a if a > 0 else 0.0.
        """
        ctx.save_for_backward(a)
        return float(operators.relu(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Backprop for ReLU.

        Args:
            ctx: Context containing saved input a.
            d_output: Upstream gradient.

        Returns:
            d_output if a > 0 else 0.0.
        """
        (a,) = ctx.saved_values
        # d/d(a) ReLU(a) = 1 if a>0 else 0 -> relu_back(a, d)
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    "Exp function"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """
        Compute exp(a).

        Saves the output e = exp(a) for the backward pass.

        Args:
            ctx: Context used to save e.
            a: Input value.

        Returns:
            exp(a).
        """
        e = operators.exp(a)
        ctx.save_for_backward(e)
        return float(e)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """
        Backprop for exp.

        Args:
            ctx: Context containing saved forward output e = exp(a).
            d_output: Upstream gradient.

        Returns:
            d_output * e.
        """
        (e,) = ctx.saved_values
        # d/d(a) exp(a) = exp(a)
        return d_output * e


class LT(ScalarFunction):
    """
    Less-than comparison returning a float indicator.

    f(x, y) = 1.0 if x < y else 0.0
    """

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """
        Compute 1.0 if a < b else 0.0.

        Saves inputs for an approximate/backstop backward near a == b.

        Args:
            ctx: Context used to save (a, b).
            a: Left operand.
            b: Right operand.

        Returns:
            1.0 if a < b else 0.0.
        """
        ctx.save_for_backward(a, b)
        return 1.0 if operators.lt(a, b) else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """
        Backprop for less-than.

        Note:
            The indicator is nondifferentiable; we return zeros almost
            everywhere, and use a tiny symmetric finite-difference
            approximation only when |a - b| is extremely small.

        Args:
            ctx: Context containing (a, b).
            d_output: Upstream gradient.

        Returns:
            (d_da, d_db) as described above.
        """
        a, b = ctx.saved_values
        eps = 1e-6  # тот же шаг, что в central_difference
        if abs(a - b) < eps:
            scale = 1.0 / (2.0 * eps)
            return -d_output * scale, d_output * scale
        return 0.0, 0.0


class EQ(ScalarFunction):
    """
    Equality comparison returning a float indicator.

    f(x, y) = 1.0 if x == y else 0.0
    """
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """
        Compute 1.0 if a == b else 0.0.

        Args:
            ctx: Context (unused).
            a: Left operand.
            b: Right operand.

        Returns:
            1.0 if a == b else 0.0.
        """
        return 1.0 if operators.eq(a, b) else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """
        Backprop for equality.

        Note:
            Equality is nondifferentiable almost everywhere for real inputs.
            We return zeros.

        Args:
            ctx: Context (unused).
            d_output: Upstream gradient.

        Returns:
            (0.0, 0.0).
        """
        return 0.0, 0.0
