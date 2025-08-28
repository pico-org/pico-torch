# Tensor module initialization
from .ops import Tensor, empty, Random
from .activation import *
from ._backward import _backward

__all__ = ["Tensor", "empty", "Random", "_backward"]
