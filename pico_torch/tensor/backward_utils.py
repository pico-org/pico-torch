import jax.numpy as jnp
from jax import jit

@jit
def _backward_4_add(grad):
    return grad, grad

@jit
def _backward_4_mul(x, y, grad):
    return grad * y, grad * x

@jit
def _backward_4_sub(grad):
    return grad, -grad
