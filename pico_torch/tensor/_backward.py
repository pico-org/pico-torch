from .ops import Tensor
import jax.numpy as jnp
from .backward_utils import (_backward_4_add,_backward_4_mul,_backward_4_sub,_backward_tanh,_backward_ReLU,_backward_ELU,
                             _backward_HardShrink,_backward_Hardsigmoid,_backward_Hardtanh,_backward_Hardswish
                             ,_backward_LeakyReLU,_backward_LogSigmoid,_backward_ReLU6,_backward_Sigmoid,
                             _backward_Softshrink,_backward_Tanhshrink,_backward_4_div,_backward_div_scalar,
                             _backward_rdiv_scalar,_backward_pow
                             )


def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper


@add_to_class(Tensor)
def _backward(self, grad=0.3):
    def safe_add_grad(parent, gradient):
        """Safely add gradient to parent if it has requires_grad=True"""
        if parent._grad is not None:
            parent._grad += gradient

    if self._ops == "+":  
        dX, dY = _backward_4_add(grad)
        safe_add_grad(self._parents[0], dX)
        safe_add_grad(self._parents[1], dY)

    if self._ops == "*":  
        x = jnp.array(self._parents[0].data)
        y = jnp.array(self._parents[1].data)
        dX, dY = _backward_4_mul(x, y, grad)
        safe_add_grad(self._parents[0], dX)
        safe_add_grad(self._parents[1], dY)

    if self._ops == "-": 
        dX, dY = _backward_4_sub(grad)
        safe_add_grad(self._parents[0], dX)
        safe_add_grad(self._parents[1], dY)

    if self._ops == "sub_scalar": 
        safe_add_grad(self._parents[0], grad)

    if self._ops == "rsub_scalar":  
        safe_add_grad(self._parents[0], -grad)

    if self._ops == "neg":
        safe_add_grad(self._parents[0], -grad)
    
    if self._ops == "add_scalar":
        safe_add_grad(self._parents[0], grad)
    
    if self._ops == "mul_scalar":
        safe_add_grad(self._parents[0], grad * self._scalar)

    if self._ops == "/":
        x = jnp.array(self._parents[0].data)
        y = jnp.array(self._parents[1].data)
        dX, dY = _backward_4_div(x, y, grad)
        safe_add_grad(self._parents[0], dX)
        safe_add_grad(self._parents[1], dY)

    if self._ops == "div_scalar":
        dX = _backward_div_scalar(self._scalar, grad)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "rdiv_scalar": 
        x = jnp.array(self._parents[0].data)
        dX = _backward_rdiv_scalar(x, self._scalar, grad)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "pow":
        x = jnp.array(self._parents[0].data)
        dX = _backward_pow(x, self._power, grad)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "Tanh":
        y = jnp.array(self.data)
        dX = _backward_tanh(y,grad)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "ReLU":
        y =  jnp.array(self.data)
        dX = _backward_ReLU(y,grad)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "ELU":
        y = jnp.array(self._parents[0].data)
        dX = _backward_ELU(y,grad,self.alpha)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "HardShrink":
        y = jnp.array(self._parents[0].data)
        dX = _backward_HardShrink(y,grad,self.lambd)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "Hardsigmoid":
        y = jnp.array(self._parents[0].data)
        dX = _backward_Hardsigmoid(y,grad)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "Hardtanh":
        y = jnp.array(self._parents[0].data)
        dX = _backward_Hardtanh(y,grad,self.min_val,self.max_val)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "Hardswish":
        y = jnp.array(self._parents[0].data)
        dX = _backward_Hardswish(y,grad)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "LeakyReLU":
        y = jnp.array(self._parents[0].data)
        dX = _backward_LeakyReLU(y,grad,self.neg_slope)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "LogSigmoid":
        y = jnp.array(self._parents[0].data)
        dX = _backward_LogSigmoid(y,grad)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "ReLU6":
        y = jnp.array(self._parents[0].data)
        dX = _backward_ReLU6(y,grad)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "Sigmoid":
        y = jnp.array(self.data)
        dX = _backward_Sigmoid(y,grad)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "Softshrink":
        y = jnp.array(self.data)
        dX = _backward_Softshrink(y,grad,self.lambd)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "Tanhshrink":
        y = jnp.array(self.data)
        dX = _backward_Tanhshrink(y,grad)
        safe_add_grad(self._parents[0], dX)

    if self._ops == "matmul":

        A = jnp.array(self._parents[0].data)
        B = jnp.array(self._parents[1].data)
        
        grad_array = jnp.array(grad)
        
        # Ensure grad_array has correct dimensions for matrix multiplication
        if grad_array.ndim == 0:
            grad_array = jnp.array([[grad_array]])
        elif grad_array.ndim == 1:
            grad_array = grad_array.reshape(-1, 1) if len(grad_array) == A.shape[0] else grad_array.reshape(1, -1)
        
        # Compute gradients with proper shape handling
        try:
            dA = grad_array @ B.T
            dB = A.T @ grad_array
        except Exception:
            # Fallback: reshape grad_array to match expected output shape
            expected_shape = (A.shape[0], B.shape[1])
            if grad_array.shape != expected_shape:
                grad_array = grad_array.reshape(expected_shape)
            dA = grad_array @ B.T
            dB = A.T @ grad_array
            
        safe_add_grad(self._parents[0], dA)
        safe_add_grad(self._parents[1], dB)


    for parent in self._parents:
        if parent.requires_grad and hasattr(parent, '_backward'):
            if parent._grad is not None:
                parent._backward(jnp.sum(parent._grad))

        