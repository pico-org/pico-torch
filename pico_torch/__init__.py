from .tensor import Tensor
import jax.numpy as jnp  # type: ignore
from jax import jit  # type: ignore

from .utils import jax_tanh,jax_ReLU
__all__ = ["Tensor"]

# Warmup JIT for scalar and typical tensor shape
jax_tanh(2)  # scalar warmup
jax_tanh(jnp.ones((3,)))  # vector warmup 

import builtins
builtins.Tensor = Tensor


def tanh(X):
    if isinstance(X, Tensor):
        array = jnp.array(X.data)
        data = jax_tanh(array)
        return Tensor(data, _parents=[X])
    else:
        raise TypeError("Expected a Tensor as input")


def ReLU(X):
    if isinstance(X,Tensor):
        array = jnp.array(X.data)
        data = jax_ReLU(array)
        return Tensor(data,_parents = [X])
    else:
        raise TypeError("Expected a Tensor as input")

