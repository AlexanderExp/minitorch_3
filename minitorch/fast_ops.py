from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations

def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        size = int(np.prod(out_shape))

        # Самый быстрый путь: формы и страйды совпадают — одна и та же позиция
        aligned = True
        if len(out_shape) != len(in_shape):
            aligned = False
        else:
            for d in range(len(out_shape)):
                if out_shape[d] != in_shape[d] or out_strides[d] != in_strides[d]:
                    aligned = False
                    break

        if aligned:
            for ord_ in prange(size):
                out_index = np.empty(MAX_DIMS, dtype=np.int32)
                # Декодируем ord_ -> out_index (без вызова to_index)
                rem = int(ord_)
                for i in range(len(out_shape) - 1, -1, -1):
                    dim = int(out_shape[i])
                    out_index[i] = rem % dim
                    rem //= dim
                pos = index_to_position(out_index, out_strides)
                out[pos] = fn(float(in_storage[pos]))
            return

        # Случай с broadcasting
        for ord_ in prange(size):
            out_index = np.empty(MAX_DIMS, dtype=np.int32)
            in_index = np.empty(MAX_DIMS, dtype=np.int32)

            rem = int(ord_)
            for i in range(len(out_shape) - 1, -1, -1):
                dim = int(out_shape[i])
                out_index[i] = rem % dim
                rem //= dim

            broadcast_index(out_index, out_shape, in_shape, in_index)
            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index, in_strides)

            out[out_pos] = fn(float(in_storage[in_pos]))

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape,
        Strides, Storage, Shape, Strides], None
]:
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
        size = int(np.prod(out_shape))

        # Быстрый путь: все три тензора строго выровнены
        aligned = True
        if not (len(out_shape) == len(a_shape) == len(b_shape)):
            aligned = False
        else:
            for d in range(len(out_shape)):
                if (
                    out_shape[d] != a_shape[d]
                    or out_shape[d] != b_shape[d]
                    or out_strides[d] != a_strides[d]
                    or out_strides[d] != b_strides[d]
                ):
                    aligned = False
                    break

        if aligned:
            for ord_ in prange(size):
                out_index = np.empty(MAX_DIMS, dtype=np.int32)
                rem = int(ord_)
                for i in range(len(out_shape) - 1, -1, -1):
                    dim = int(out_shape[i])
                    out_index[i] = rem % dim
                    rem //= dim
                pos = index_to_position(out_index, out_strides)
                out[pos] = fn(float(a_storage[pos]), float(b_storage[pos]))
            return

        # broadcasting для a и b
        for ord_ in prange(size):
            out_index = np.empty(MAX_DIMS, dtype=np.int32)
            a_index = np.empty(MAX_DIMS, dtype=np.int32)
            b_index = np.empty(MAX_DIMS, dtype=np.int32)

            rem = int(ord_)
            for i in range(len(out_shape) - 1, -1, -1):
                dim = int(out_shape[i])
                out_index[i] = rem % dim
                rem //= dim

            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index, a_strides)
            b_pos = index_to_position(b_index, b_strides)

            out[out_pos] = fn(float(a_storage[a_pos]), float(b_storage[b_pos]))

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_size = int(np.prod(out_shape))
        red_len = int(a_shape[reduce_dim])
        red_stride = int(a_strides[reduce_dim])

        for ord_ in prange(out_size):
            out_index = np.empty(MAX_DIMS, dtype=np.int32)
            a_index = np.empty(MAX_DIMS, dtype=np.int32)

            # ord_ -> out_index
            rem = int(ord_)
            for i in range(len(out_shape) - 1, -1, -1):
                dim = int(out_shape[i])
                out_index[i] = rem % dim
                rem //= dim

            out_pos = index_to_position(out_index, out_strides)

            for d in range(len(a_index)):
                a_index[d] = out_index[d]
            a_index[reduce_dim] = 0
            a_base = index_to_position(a_index, a_strides)

            acc = float(out[out_pos])
            pos = a_base
            for _ in range(red_len):
                acc = fn(acc, float(a_storage[pos]))
                pos += red_stride

            out[out_pos] = acc

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
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
    """
    NUMBA tensor matrix multiply function.

    Assumes 3D tensors with optional broadcast along batch dim 0:
      a: (N_a or 1, I, K)
      b: (N_b or 1, K, J)
      out: (N = broadcast(N_a, N_b), I, J)

    Optimizations:
      * Outer loop in parallel (over all out elements)
      * No index buffers or helper function calls
      * Inner loop has no global writes and exactly 1 multiply per iteration
    """
    # batch broadcasting strides (0 если размер 1)
    a_batch_stride = int(a_strides[0]) if int(a_shape[0]) > 1 else 0
    b_batch_stride = int(b_strides[0]) if int(b_shape[0]) > 1 else 0

    # размеры
    N = int(out_shape[0])        # batch
    I = int(out_shape[1])        # rows
    J = int(out_shape[2])        # cols
    K = int(a_shape[2])          # shared dim (a: ..., K), (b: K, ...)

    # страйды как int (во избежание float-индексов)
    os0, os1, os2 = int(out_strides[0]), int(
        out_strides[1]), int(out_strides[2])
    as1, as2 = int(a_strides[1]), int(a_strides[2])
    bs1, bs2 = int(b_strides[1]), int(b_strides[2])

    total = N * I * J

    # Внешний цикл — параллельно
    for ord_ in prange(total):
        # Распаковать ord_ -> (n, i, j) без буферов и вызовов
        n = ord_ // (I * J)
        rem = ord_ - n * (I * J)
        i = rem // J
        j = rem - i * J

        # Позиция в out
        out_pos = n * os0 + i * os1 + j * os2

        # Базовые позиции (для k = 0)
        a_base = n * a_batch_stride + i * as1           # + 0 * as2
        b_base = n * b_batch_stride + j * bs2           # + 0 * bs1

        # Аккумуляция локально; глобальная запись ровно одна — в конце
        acc = 0.0
        a_pos = a_base
        b_pos = b_base
        for _k in range(K):
            acc += float(a_storage[a_pos]) * \
                float(b_storage[b_pos])  # 1 умножение
            a_pos += as2
            b_pos += bs1

        out[out_pos] = acc



tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
