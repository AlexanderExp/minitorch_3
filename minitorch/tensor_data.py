from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Convert a multi-dimensional index into a 1D storage position.

    The mapping uses row-major (C-style) layout encoded by `strides`.
    For each dimension `i`, the linear position adds `index[i] * strides[i]`.

    Args:
        index: N-D integer index (length equals the number of dims).
        strides: N-D strides describing the storage layout.

    Returns:
        int: The single-dimensional position in the underlying storage.
    """

    pos = 0
    for i in range(len(strides)):
        pos += int(index[i]) * int(strides[i])
    return int(pos)


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert a linear ordinal into a multi-dimensional index for `shape`.

    Enumerating ordinals from `0` to `prod(shape)-1` will enumerate each
    index in the tensor exactly once (row-major order). Note this is not
    necessarily the inverse of `index_to_position` for arbitrary strides.

    Args:
        ordinal: Linear position in [0, prod(shape)).
        shape: Tensor shape array (length = number of dims).
        out_index: Output array to be filled in-place with the N-D index.

    Returns:
        None
    """
    # Walk from the last dimension to the first; modulo/division extracts
    # the coordinate on that axis in row-major enumeration.
    for i in range(len(shape) - 1, -1, -1):
        dim = int(shape[i])
        out_index[i] = int(ordinal % dim)
        ordinal //= dim


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """
    Map an index from a broadcasted (bigger) shape to a smaller shape.

    Follows NumPy-style broadcasting rules: if a dimension of `shape`
    equals 1, the corresponding output index is 0 for that dimension;
    otherwise it is copied from the aligned `big_index`.

    Args:
        big_index: Index in the broadcasted (larger) shape.
        big_shape: The broadcasted (larger) tensor shape.
        shape: The target (smaller) tensor shape.
        out_index: Output array to write the mapped index (in-place).

    Returns:
        None
    """
    n_big = int(len(big_shape))
    n_small = int(len(shape))
    offset = n_big - n_small

    # Iterate from right to left; align trailing dimensions, leading
    # unmatched big dims are effectively ignored (treated as size 1).
    for i in range(n_small - 1, -1, -1):
        # индекс в больших осях, если такой есть, иначе считаем "виртуальную" ось размером 1
        big_i = i + offset
        idx_from_big = int(big_index[big_i]) if big_i >= 0 else 0

        # If target dim is broadcasted (== 1), index is 0; else copy aligned idx
        if int(shape[i]) == 1:
            out_index[i] = 0
        else:
            out_index[i] = idx_from_big


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """
    Broadcast two shapes to a common shape using NumPy rules.

    The two shapes are aligned on the right; dimensions must be equal or
    one of them must be 1 on every axis.

    Args:
        shape1: First shape (sequence of positive ints).
        shape2: Second shape (sequence of positive ints).

    Returns:
        tuple[int, ...]: The broadcasted shape.

    Raises:
        IndexingError: If the shapes cannot be broadcast together.
    """
    s1 = tuple(int(x) for x in shape1)
    s2 = tuple(int(x) for x in shape2)
    n = max(len(s1), len(s2))

    # Left-pad the shorter shape with ones.
    a = (1,) * (n - len(s1)) + s1
    b = (1,) * (n - len(s2)) + s2

    out: list[int] = []
    for d1, d2 in zip(a, b):
        if d1 == d2 or d1 == 1 or d2 == 1:
            out.append(max(d1, d2))
        else:
            raise IndexingError(f"Cannot broadcast shapes {shape1} and {shape2}.")
    return tuple(out)


def strides_from_shape(shape: UserShape) -> UserStrides:
    """
    Compute row-major (C-style) strides for a given shape.

    For shape (s0, s1, ..., s_{n-1}), the strides are:
        (s1*s2*...*s_{n-1}, s2*...*s_{n-1}, ..., s_{n-1}, 1)

    Args:
        shape: Tensor shape as an iterable of ints.

    Returns:
        tuple[int, ...]: Strides matching row-major layout.
    """
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    """
    Low-level tensor storage, shape, and stride management.

    This class provides:
      * Validation and storage of raw data buffer (`_storage`).
      * Row-major strides or custom strides (`_strides`).
      * Indexing utilities (`index`, `indices`, `get`, `set`).
      * Layout transforms such as `permute`.
      * CUDA transfer helper `to_cuda_` (optional).

    Note:
        `TensorData` does not implement user-facing tensor semantics; it is
        the backend for `minitorch.Tensor`.
    """
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            order (list): a permutation of the dimensions

        Returns:
            New `TensorData` with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        new_shape = tuple(self.shape[i] for i in order)
        new_strides = tuple(self.strides[i] for i in order)
        return TensorData(self._storage, new_shape, new_strides)

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
