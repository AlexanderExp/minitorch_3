from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        pass

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`


        Returns :
            A collection of tensor functions

        """

        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map.

        Simple version:
            out[i, j, ...] = fn(a[i, j, ...])

        Broadcasted version:
            out[i, j, ...] = fn(a[bi, bj, ...]) where ``bi``/``bj`` follow
            standard broadcasting rules.

        Args:
            fn: Scalar function to apply.

        Returns:
            A callable that applies ``fn`` elementwise with broadcasting.
        """

        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip.

        Simple version:
            out[i, j, ...] = fn(a[i, j, ...], b[i, j, ...])

        Broadcasted version:
            out[i, j, ...] = fn(a[bi, bj, ...], b[ci, cj, ...])

        Args:
            fn: Binary scalar function to apply.

        Returns:
            A callable that zips two tensors into a new output tensor.
        """

        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Higher-order tensor reduce along a single dimension.

        Simple version (reduce over ``dim``):
            out[..., 1, ...] = fold(fn, start, a[..., i, ...] over i in dim)

        Args:
            fn: Associative binary reduction function.
            start: Initial value written to the output buffer before folding.

        Returns:
            A callable that reduces the input tensor along the given dimension.
        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(fn: Callable[[float], float]) -> Any:
    """Low-level strided/broadcasted map.

    Fills the ``out`` storage by applying ``fn`` to the broadcast-aligned
    element from ``in_storage`` for each output index.

    Args:
        fn: Scalar function to apply.
        out: Output storage.
        out_shape: Output shape.
        out_strides: Output strides.
        in_storage: Input storage.
        in_shape: Input shape.
        in_strides: Input strides.

    Returns:
        A callable that writes in-place into ``out``.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # Идём по всем индексам out, через broadcast сопоставляя индекс входа.
        size = int(np.prod(out_shape))
        out_index = np.zeros_like(out_shape, dtype=np.int32)
        in_index = np.zeros_like(in_shape, dtype=np.int32)

        for ord_ in range(size):
            to_index(ord_, out_shape, out_index)
            # сопоставить out_index -> in_index через broadcasting
            broadcast_index(out_index, out_shape, in_shape, in_index)

            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)

            out[out_pos] = fn(float(in_storage[in_pos]))

    return _map


def tensor_zip(fn: Callable[[float, float], float]) -> Any:
    """Low-level strided/broadcasted zip of two tensors.

    Fills the ``out`` storage by applying ``fn`` to broadcast-aligned elements
    from ``a_storage`` and ``b_storage`` for each output index.

    Args:
        fn: Binary scalar function to apply.
        out: Output storage.
        out_shape: Output shape.
        out_strides: Output strides.
        a_storage: First input storage.
        a_shape: First input shape.
        a_strides: First input strides.
        b_storage: Second input storage.
        b_shape: Second input shape.
        b_strides: Second input strides.

    Returns:
        A callable that writes in-place into ``out``.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # Идём по всем индексам out, сопоставляя индекс a и b через broadcasting
        size = int(np.prod(out_shape))
        out_index = np.zeros_like(out_shape, dtype=np.int32)
        a_index = np.zeros_like(a_shape, dtype=np.int32)
        b_index = np.zeros_like(b_shape, dtype=np.int32)

        for ord_ in range(size):
            to_index(ord_, out_shape, out_index)

            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            out[out_pos] = fn(float(a_storage[a_pos]), float(b_storage[b_pos]))

    return _zip


def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
    """Low-level reduction along a single dimension.

    The output shape must equal the input shape except that the reduced
    dimension has size ``1``.

    Args:
        fn: Associative binary reduction function.
        out: Output storage.
        out_shape: Output shape (with reduced dim set to 1).
        out_strides: Output strides.
        a_storage: Input storage to reduce.
        a_shape: Input shape.
        a_strides: Input strides.
        reduce_dim: Dimension index to reduce over.

    Returns:
        A callable that writes in-place into ``out``.
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # out_shape = a_shape с reduce_dim == 1
        # Для каждого out-индекса пробегаем все значения вдоль reduce_dim во входе.
        out_size = int(np.prod(out_shape))
        out_index = np.zeros_like(out_shape, dtype=np.int32)
        a_index = np.zeros_like(a_shape, dtype=np.int32)

        for ord_ in range(out_size):
            to_index(ord_, out_shape, out_index)

            out_pos = index_to_position(out_index, out_strides)
            acc = float(out[out_pos])

            # Скопировать out_index в a_index
            for d in range(len(a_shape)):
                a_index[d] = out_index[d]
            for r in range(int(a_shape[reduce_dim])):
                a_index[reduce_dim] = r
                a_pos = index_to_position(a_index, a_strides)
                acc = fn(acc, float(a_storage[a_pos]))

            out[out_pos] = acc

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
