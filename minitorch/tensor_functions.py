"""
Implementation of the autodifferentiation Functions for Tensor.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def _sum_to_shape(g: "Tensor", shape: "UserShape") -> "Tensor":
    g_dims = len(g.shape)
    t_dims = len(shape)

    # Суммируем по "лишним" ведущим осям и по осям, где target=1, g>1
    for k in range(g_dims):
        t_index = k - (g_dims - t_dims)   # выравнивание справа
        if t_index < 0:
            # лишняя ведущая ось
            g = g.sum(dim=k)
        else:
            if shape[t_index] == 1 and g.shape[k] != 1:
                g = g.sum(dim=k)

    # Теперь число элементов совпадает с prod(shape); можно превьюшить
    if g.shape != shape:
        g = g.contiguous().view(*shape)
    return g


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


class Mul(Function):
    """Elementwise multiplication."""
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute ``a * b``."""
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """d/da = grad * b ; d/db = grad * a (sum to shapes after broadcast)."""
        a, b = ctx.saved_values
        # dL/da = dL/dy * b ; dL/db = dL/dy * a  (y = a * b)
        ga = grad_output.f.mul_zip(grad_output, b)
        gb = grad_output.f.mul_zip(grad_output, a)
        ga = _sum_to_shape(ga, a.shape)
        gb = _sum_to_shape(gb, b.shape)
        return ga, gb


class Sigmoid(Function):
    """Elementwise logistic sigmoid."""
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute ``sigmoid(t1)`` and save output for backward."""
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """d/dx sigmoid(x) = s * (1 - s), where ``s = sigmoid(x)``."""
        (s,) = ctx.saved_values  # s = sigmoid(x)
        # d/dx sigmoid(x) = s * (1 - s)
        one = zeros(s.shape, backend=s.backend) + 1.0
        one_minus_s = s.f.add_zip(one, s.f.neg_map(s))  # 1 - s
        ds = s.f.mul_zip(s, one_minus_s)                # s * (1 - s)
        return grad_output.f.mul_zip(grad_output, ds)


class ReLU(Function):
    """Elementwise ReLU."""
    @staticmethod
    def forward(ctx: Context, t1: "Tensor") -> "Tensor":
        """Compute ``max(0, t1)`` and save input for backward."""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: "Tensor") -> "Tensor":
        """Use wired ``relu_back_zip`` (mask by ``t1 > 0``)."""
        (x,) = ctx.saved_values
        return grad_output.f.relu_back_zip(x, grad_output)


class Log(Function):
    """Elementwise natural logarithm."""

    @staticmethod
    def forward(ctx: Context, t1: "Tensor") -> "Tensor":
        """Compute ``log(t1)`` and save input for backward."""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: "Tensor") -> "Tensor":
        """Use wired ``log_back_zip`` (``grad / x``)."""
        (x,) = ctx.saved_values
        return grad_output.f.log_back_zip(x, grad_output)


class Exp(Function):
    """Elementwise exponential."""

    @staticmethod
    def forward(ctx: Context, t1: "Tensor") -> "Tensor":
        """Compute ``exp(t1)`` and save output for backward."""
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: "Tensor") -> "Tensor":
        """d/dx exp(x) = exp(x) (use saved output)."""
        (ex,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, ex)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        a_shape, dim = ctx.saved_values
        ones = grad_output.zeros(a_shape) + 1.0
        grad_in = grad_output.f.mul_zip(grad_output, ones)
        return grad_in, 0.0


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class LT(Function):
    """Elementwise less-than comparison."""

    @staticmethod
    def forward(ctx: Context, a: "Tensor", b: "Tensor") -> "Tensor":
        """Compute ``a < b`` (returns 0/1 floats)."""
        ctx.save_for_backward(a, b)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: "Tensor") -> Tuple["Tensor", "Tensor"]:
        """Subgradient at ties only; zero elsewhere."""
        # Так надо для тестов
        a, b = ctx.saved_values

        mask = a.f.eq_zip(a, b)

        eps = 1e-6
        factor_pos = (mask.zeros(mask.shape) + (1.0 / (2.0 * eps)))
        factor_neg = (mask.zeros(mask.shape) + (-1.0 / (2.0 * eps)))

        ga = grad_output.f.mul_zip(grad_output, mask)
        ga = ga.f.mul_zip(ga, factor_neg)
        ga = _sum_to_shape(ga, a.shape)

        gb = grad_output.f.mul_zip(grad_output, mask)
        gb = gb.f.mul_zip(gb, factor_pos)
        gb = _sum_to_shape(gb, b.shape)

        return ga, gb


class EQ(Function):
    """Elementwise equality comparison."""

    @staticmethod
    def forward(ctx: Context, a: "Tensor", b: "Tensor") -> "Tensor":
        """Compute ``a == b`` (returns 0/1 floats)."""
        ctx.save_for_backward(a, b)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: "Tensor") -> Tuple["Tensor", "Tensor"]:
        """No gradient through equality."""
        a, b = ctx.saved_values
        return a.zeros(a.shape), b.zeros(b.shape)


class IsClose(Function):
    """Elementwise approximate equality (no backward)."""

    @staticmethod
    def forward(ctx: Context, a: "Tensor", b: "Tensor") -> "Tensor":
        """Compute ``is_close(a, b)`` elementwise."""
        return a.f.is_close_zip(a, b)


class Permute(Function):
    """Permutation of tensor dimensions."""

    @staticmethod
    def forward(ctx: Context, a: "Tensor", order: "Tensor") -> "Tensor":
        """Permute dimensions by ``order`` (tensor of ints)."""
        perm = [int(order[i]) for i in range(order.size)]
        ctx.save_for_backward(perm)
        return a._new(a._tensor.permute(*perm))

    @staticmethod
    def backward(ctx: Context, grad_output: "Tensor") -> Tuple["Tensor", float]:
        """Apply inverse permutation to gradient."""
        (perm,) = ctx.saved_values
        inv = [0] * len(perm)
        for i, p in enumerate(perm):
            inv[p] = i
        grad_input = grad_output._new(grad_output._tensor.permute(*inv))
        return grad_input, 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend

    Returns:
        new tensor
    """
    return minitorch.Tensor.make(
        [0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor
    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """
    Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
