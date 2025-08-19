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
    if self._ops == "+":  
        dX, dY = _backward_4_add(grad)
        self._parents[0]._grad += dX
        self._parents[1]._grad += dY

    if self._ops == "*":  
        x = jnp.array(self._parents[0].data)
        y = jnp.array(self._parents[1].data)
        dX, dY = _backward_4_mul(x, y, grad)
        self._parents[0]._grad += dX
        self._parents[1]._grad += dY

    if self._ops == "-": 
        dX, dY = _backward_4_sub(grad)
        self._parents[0]._grad += dX
        self._parents[1]._grad += dY  

    if self._ops == "sub_scalar": 
        self._parents[0]._grad += grad

    if self._ops == "rsub_scalar":  
        self._parents[0]._grad += -grad

    if self._ops == "neg":
        self._parents[0]._grad += -grad
    
    if self._ops == "add_scalar":
        self._parents[0]._grad += grad
    
    if self._ops == "mul_scalar":
        self._parents[0]._grad += grad * self._scalar

    if self._ops == "/":
        if len(self._parents) == 2:  
            x = jnp.array(self._parents[0].data)
            y = jnp.array(self._parents[1].data)
            dX, dY = _backward_4_div(x, y, grad)
            self._parents[0]._grad += dX
            self._parents[1]._grad += dY
        else:  # tensor / scalar case
            dX = _backward_div_scalar(self._scalar, grad)
            self._parents[0]._grad += dX

    if self._ops == "rdiv_scalar": 
        x = jnp.array(self._parents[0].data)
        dX = _backward_rdiv_scalar(x, self._scalar, grad)
        self._parents[0]._grad += dX

    if self._ops == "pow":
        x = jnp.array(self._parents[0].data)
        dX = _backward_pow(x, self._power, grad)
        self._parents[0]._grad += dX


    if self._ops == "Tanh":
        y = jnp.array(self.data)
        dX = _backward_tanh(y,grad)
        self._parents[0]._grad += dX

    if self._ops == "ReLU":
        y =  jnp.array(self.data)
        dX = _backward_ReLU(y,grad)
        self._parents[0]._grad += dX

    if self._ops == "ELU":
        y = jnp.array(self._parents[0].data)
        dX = _backward_ELU(y,grad,self.alpha)
        self._parents[0]._grad += dX

    if self._ops == "HardShrink":
        y = jnp.array(self._parents[0].data)
        dX = _backward_HardShrink(y,grad,self.lambd)
        self._parents[0]._grad += dX

    if self._ops == "Hardsigmoid":
        y = jnp.array(self._parents[0].data)
        dX = _backward_Hardsigmoid(y,grad)
        self._parents[0]._grad += dX

    if self._ops == "Hardtanh":
        y = jnp.array(self._parents[0].data)
        dX = _backward_Hardtanh(y,grad,self.min_val,self.max_val)
        self._parents[0]._grad += dX

    if self._ops == "Hardswish":
        y = jnp.array(self._parents[0].data)
        dX = _backward_Hardswish(y,grad)
        self._parents[0]._grad += dX

    if self._ops == "LeakyReLU":
        y = jnp.array(self._parents[0].data)
        dX = _backward_LeakyReLU(y,grad,self.neg_slope)
        self._parents[0]._grad += dX

    if self._ops == "LogSigmoid":
        y = jnp.array(self._parents[0].data)
        dX = _backward_LogSigmoid(y,grad)
        self._parents[0]._grad += dX

    if self._ops == "ReLU6":
        y = jnp.array(self._parents[0].data)
        dX = _backward_ReLU6(y,grad)
        self._parents[0]._grad += dX

    if self._ops == "Sigmoid":
        y = jnp.array(self.data)
        dX = _backward_Sigmoid(y,grad)
        self._parents[0]._grad += dX

    if self._ops == "Softshrink":
        y = jnp.array(self.data)
        dX = _backward_Softshrink(y,grad,self.lambd)
        self._parents[0]._grad += dX

    if self._ops == "Tanhshrink":
        y = jnp.array(self.data)
        dX = _backward_Tanhshrink(y,grad)
        self._parents[0]._grad += dX




        