import jax  
from jax import jit 
import jax.numpy as jnp 

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
    return jnp.where(X > max_val, max_val,jnp.where(X<min_val, min_val, X))  # type: ignore


@jit 
def jax_Hardswish(X):
    return jnp.where(X >= 3, X, jnp.where(X <= -3, 0, (X*((X+3)/6)))) # type: ignore

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
    return jnp.where(X > lambd,X - lambd, jnp.where(X < -lambd, X + lambd, 0)) # type: ignore

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

def _initialize_activation_functions():
    import jax.numpy as jnp
    dummy_x = jnp.array([1.0])
    dummy_alpha = 1.0
    dummy_lambd = 0.5
    dummy_min_val = -1.0
    dummy_max_val = 1.0
    dummy_neg_slope = 0.1
    jax_tanh(dummy_x)
    jax_ReLU(dummy_x)
    jax_ELU(dummy_x, dummy_alpha)
    jax_HardShrink(dummy_x, dummy_lambd)
    jax_Hardsigmoid(dummy_x)
    jax_Hardtanh(dummy_x, dummy_min_val, dummy_max_val)
    jax_Hardswish(dummy_x)
    jax_LeakyReLU(dummy_x, dummy_neg_slope)
    jax_LogSigmoid(dummy_x)
    jax_ReLU6(dummy_x)
    jax_Sigmoid(dummy_x)
    jax_Softshrink(dummy_x, dummy_lambd)
    jax_Softsign(dummy_x)
    jax_Tanhshrink(dummy_x)
    jax_Softmax(dummy_x)
    jax_LogSoftmax(dummy_x)