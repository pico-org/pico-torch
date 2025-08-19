import jax.numpy as jnp
from jax import jit
from .activation_utils import jax_Sigmoid
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
    return grad * (1 - y**2)


@jit 
def _backward_ReLU(y,grad):
    return grad * jnp.where(y > 0, 1, 0)


@jit 
def _backward_ELU(y,grad,alpha):
    return grad * jnp.where(y > 0, 1, alpha * jnp.exp(y))


@jit 
def _backward_HardShrink(y,grad,lambd):
    return grad * jnp.where(jnp.abs(y) > lambd, 1, 0)


@jit
def _backward_Hardsigmoid(y,grad):
    return grad * jnp.where(jnp.abs(y) >= 3, 0, 1/6)


@jit
def _backward_Hardtanh(y,grad,min_val,max_val):
    return grad * jnp.where(y > max_val, 0, jnp.where(y < min_val, 0, 1))

@jit 
def _backward_Hardswish(y,grad):
    return grad * jnp.where(y >= 3, 1, jnp.where(y <= -3, 0, 1/3))

@jit 
def _backward_LeakyReLU(y,grad,neg_slope):
    return grad * jnp.where(y >= 0, 1, neg_slope * 1)

@jit
def _backward_LogSigmoid(y,grad):
    return grad * jax_Sigmoid(y)

@jit
def _backward_ReLU6(y,grad):
    return grad * jnp.where(y < 0, 0, jnp.where(y < 6, 1, 0))

@jit
def _backward_Sigmoid(y,grad):
    return grad * (y * (1 - y))


@jit
def _backward_Softshrink(y,grad,lambd):
    return grad * jnp.where(y > lambd, 1, jnp.where(y < -lambd, 1, 0))


@jit 
def _backward_Tanhshrink(y,grad):
    return grad * (1 - y**2)

@jit
def _backward_4_div(x, y, grad):
    return grad / y, grad * (-x / (y**2))

@jit
def _backward_div_scalar(scalar, grad):
    return grad / scalar

@jit
def _backward_rdiv_scalar(x, scalar, grad):
    return grad * (-scalar / (x**2))

@jit
def _backward_pow(x, n, grad):
    return grad * n * (x**(n-1))

def _initialize_backward_functions():
    import jax.numpy as jnp
    dummy_grad = jnp.array([1.0])
    dummy_x = jnp.array([1.0])
    dummy_y = jnp.array([2.0])
    dummy_scalar = 2.0
    dummy_alpha = 1.0
    dummy_lambd = 0.5
    dummy_min_val = -1.0
    dummy_max_val = 1.0
    dummy_neg_slope = 0.1
    dummy_power = 2.0

    _backward_4_add(dummy_grad)
    _backward_4_mul(dummy_x, dummy_y, dummy_grad)
    _backward_4_sub(dummy_grad)
    _backward_tanh(dummy_x, dummy_grad)
    _backward_ReLU(dummy_x, dummy_grad)
    _backward_ELU(dummy_x, dummy_grad, dummy_alpha)
    _backward_HardShrink(dummy_x, dummy_grad, dummy_lambd)
    _backward_Hardsigmoid(dummy_x, dummy_grad)
    _backward_Hardtanh(dummy_x, dummy_grad, dummy_min_val, dummy_max_val)
    _backward_Hardswish(dummy_x, dummy_grad)
    _backward_LeakyReLU(dummy_x, dummy_grad, dummy_neg_slope)
    _backward_LogSigmoid(dummy_x, dummy_grad)
    _backward_ReLU6(dummy_x, dummy_grad)
    _backward_Sigmoid(dummy_x, dummy_grad)
    _backward_Softshrink(dummy_x, dummy_grad, dummy_lambd)
    _backward_Tanhshrink(dummy_x, dummy_grad)
    _backward_4_div(dummy_x, dummy_y, dummy_grad)
    _backward_div_scalar(dummy_scalar, dummy_grad)
    _backward_rdiv_scalar(dummy_x, dummy_scalar, dummy_grad)
    _backward_pow(dummy_x, dummy_power, dummy_grad)

 