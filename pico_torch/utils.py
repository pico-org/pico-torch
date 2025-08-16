import jax # type: ignore 
from jax import jit # type: ignore
import jax.numpy as jnp # type: ignore

@jit
def jax_tanh(X):
    return (jnp.exp(X) - jnp.exp(-X)) / (jnp.exp(X) + jnp.exp(-X))


@jit 
def jax_ReLU(X):
    return jnp.maximum(0,X)


@jit
def jax_ELU(X,alpha = 1.0):
    return jnp.where(X > 0, X, alpha * (jnp.exp(X) - 1))


@jit
def jax_HardShrink(X, lambd=0.5):
    return jnp.where(jnp.abs(X) >= lambd, X, 0.0)

@jit 
def jax_Hardsigmoid(X):
    return jnp.where(jnp.abs(X) >= 3, jnp.where(X >= 3, 1, 0), (X/6)+0.5)


@jit 
def jax_Hardtanh(X,min_val=-1.0, max_val=1.0):
    return jnp.where(X > max_val, max_val,jnp.where(X<min_val, min_val, X))


@jit 
def jax_Hardswish(X):
    return jnp.where(X >= 3, X, jnp.where(X <= -3, 0, (X*((X+3)/6))))

@jit 
def jax_LeakyReLU(X,negative_slope = 1e-2):
    return jnp.where(X >= 0, X, negative_slope*X)

@jit
def jax_LogSigmoid(X):
    return jnp.log(1/(1+jnp.exp(-X)))

@jit
def jax_ReLU6(X):
    return jnp.minimum(jnp.maximum(0,X),6)


@jit 
def jax_Sigmoid(X):
    return 1/(1+jnp.exp(-X))

@jit
def jax_Softshrink(X,lambd = 0.5):
    return jnp.where(X > lambd,X - lambd, jnp.where(X < -lambd, X + lambd, 0))

@jit
def jax_Softsign(X):
    return X/(1+jnp.abs(X))

@jit
def jax_Tanhshrink(X):
    return X - jnp.tanh(X)

@jit 
def jax_Softmax(X):
    return jnp.exp(X)/(1e-5+jnp.sum(jnp.exp(X)))


@jit 
def jax_LogSoftmax(X):
    return jnp.log(jnp.exp(X)/(1e-5+jnp.sum(jnp.exp(X))))