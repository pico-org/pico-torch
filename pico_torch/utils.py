import jax # type: ignore 
from jax import jit # type: ignore
import jax.numpy as jnp # type: ignore

@jit
def jax_tanh(X):
    return (jnp.exp(X) - jnp.exp(-X)) / (jnp.exp(X) + jnp.exp(-X))


@jit 
def jax_ReLU(X):
    return jnp.maximum(0,X)
