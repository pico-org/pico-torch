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

@jit 
def _backward_tanh(y,grad):
    return grad*(1-y**2)


@jit 
def _backward_ReLU(y,grad):
    jnp.where(y>0,1,0)
    return grad*y


@jit 
def _backward_ELU(y,grad,alpha):
    return grad*(jnp.where(y>0,1,alpha*jnp.exp(y)))


@jit 
def _backward_HardShrink(y,grad,lambd):
    return grad*(jnp.where(jnp.abs(y)>lambd,1,0))


@jit
def _backward_Hardsigmoid(y,grad):
    return grad*(jnp.where(jnp.abs(y)>=3,0,1/6))


@jit
def _backward_Hardtanh(y,grad,min_val,max_val):
    return grad*(jnp.where(y>max_val,0,jnp.where(y<min_val,0,1)))