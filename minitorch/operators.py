import math
from typing import Callable, Iterable, List, TypeVar, Union
# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
        x: First multiplicand.
        y: Second multiplicand.

    Returns:
        The product x * y.
    """
    return x * y


def id(input: object) -> object:
    """Identity function.

    Args:
        input: Any value.

    Returns:
        The input unchanged.
    """
    return input


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
        x: First addend.
        y: Second addend.

    Returns:
        The sum x + y.
    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
        x: Input value.

    Returns:
        The negation -x.
    """
    return -x


def lt(x: float, y: float) -> float:
    """Check if one number is less than another.

    Args:
        x: Left-hand side value.
        y: Right-hand side value.

    Returns:
        True if x < y; otherwise False.
    """
    return x < y


def eq(x: float, y: float) -> float:
    """Check if two numbers are equal.

    Args:
        x: First value.
        y: Second value.

    Returns:
        True if x == y; otherwise False.
    """
    return x == y


def max(x: float, y: float) -> float:
    """Return the larger of two numbers.

    Args:
        x: First value.
        y: Second value.

    Returns:
        x if x > y else y.
    """
    return x if x > y else y


def is_close(x: float, y: float, tol: float = 1e-2) -> bool:
    """Check if two numbers are within a given tolerance.

    Args:
        x: First value.
        y: Second value.
        tol: Absolute tolerance. Defaults to 1e-2.

    Returns:
        True if abs(x - y) < tol; otherwise False.
    """
    return abs(x - y) < tol


def sigmoid(x: float) -> float:
    """Compute the logistic sigmoid function.

    Uses a numerically stable formulation:
    - For x >= 0: 1 / (1 + exp(-x))
    - For x < 0: exp(x) / (1 + exp(x))

    Args:
        x: Input value.

    Returns:
        The sigmoid of x in (0, 1).
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Apply the ReLU activation function.

    Args:
        x: Input value.

    Returns:
        x if x > 0 else 0.0.
    """
    return x if x > 0 else 0


def log(x: float) -> float:
    """Compute the natural logarithm.

    Args:
        x: Positive input value.

    Returns:
        ln(x).
    """
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential function.

    Args:
        x: Input value.

    Returns:
        e**x.
    """
    return math.exp(x)


def inv(x: float) -> float:
    """Compute the reciprocal.

    Args:
        x: Non-zero input value.

    Returns:
        1.0 / x.
    """
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Backward pass helper for log(x).

    Computes d * d/dx log(x) = d / x.

    Args:
        x: Forward input to log (must be > 0 in forward pass).
        d: Incoming gradient from upstream.

    Returns:
        The gradient contribution d / x.
    """
    return d / x


def inv_back(x: float, d: float) -> float:
    """Backward pass helper for inv(x) where inv(x) = 1/x.

    Computes d * d/dx (1/x) = d * (-1/x^2).

    Args:
        x: Forward input to inv (must be non-zero in forward pass).
        d: Incoming gradient from upstream.

    Returns:
        The gradient contribution -d / (x * x).
    """
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Backward pass helper for relu(x) where relu(x) = max(0, x).

    The derivative is 1 when x > 0 and 0 otherwise.

    Args:
        x: Forward input to relu.
        d: Incoming gradient from upstream.

    Returns:
        d if x > 0 else 0.0.
    """
    return d if x > 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

T = TypeVar("T")
U = TypeVar("U")
Number = Union[int, float]


def map(fn: Callable[[T], U], xs: Iterable[T]) -> List[U]:  # noqa: A003
    """Apply a unary function to each element of an iterable.

    Args:
        fn: Function applied to each element.
        xs: Input iterable.

    Returns:
        A new list containing fn(x) for every x in xs.
    """
    return [fn(x) for x in xs]


def zipWith(fn: Callable[[T, U], Number], xs: Iterable[T], ys: Iterable[U]) -> List[Number]:
    """Combine two iterables element-wise with a binary function.

    Iteration stops at the end of the shorter iterable.

    Args:
        fn: Function of two arguments to combine elements.
        xs: First iterable.
        ys: Second iterable.

    Returns:
        A new list containing fn(x, y) for paired elements of xs and ys.
    """
    return [fn(x, y) for x, y in zip(xs, ys)]


def reduce(fn: Callable[[T, T], T], xs: Iterable[T]) -> T:  # noqa: A003
    """Reduce an iterable to a single value using a binary function.

    The first element is used as the initial accumulator value.

    Args:
        fn: Associative binary function used to combine elements.
        xs: Input iterable. Must be non-empty.

    Returns:
        The accumulated value.
    """
    iterator = iter(xs)
    value = next(iterator)
    for element in iterator:
        value = fn(value, element)
    return value


def negList(xs: Iterable[Number]) -> List[Number]:
    """Negate all elements in a sequence.

    Args:
        xs: Input numeric iterable.

    Returns:
        A list with each element negated.
    """
    return map(lambda x: -x, xs)


def addLists(xs: Iterable[Number], ys: Iterable[Number]) -> List[Number]:
    """Add two numeric sequences element-wise.

    Args:
        xs: First numeric iterable.
        ys: Second numeric iterable.

    Returns:
        A list where each element is x + y for paired elements.
    """
    return zipWith(lambda a, b: a + b, xs, ys)


def sum(xs: Iterable[Number]) -> Number:
    """Compute the sum of a sequence.

    Returns 0.0 for an empty iterable to mirror common reductions.

    Args:
        xs: Input numeric iterable.

    Returns:
        The arithmetic sum of elements.
    """
    items = list(xs)
    if not items:
        return 0.0
    return reduce(lambda a, b: a + b, items)


def prod(xs: Iterable[Number]) -> Number:
    """Compute the product of a sequence.

    Returns 1.0 for an empty iterable to mirror common reductions.

    Args:
        xs: Input numeric iterable.

    Returns:
        The product of all elements.
    """
    items = list(xs)
    if not items:
        return 1.0
    return reduce(lambda a, b: a * b, items)
