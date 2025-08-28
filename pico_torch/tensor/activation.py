import jax.numpy as jnp 
from .ops import Tensor

from .activation_utils import (jax_tanh,jax_ReLU, jax_ELU, jax_HardShrink, jax_Hardsigmoid,jax_Hardtanh,
                    jax_Hardswish,jax_LeakyReLU,jax_LogSigmoid,jax_ReLU6,jax_Sigmoid,jax_Softshrink,
                    jax_Softsign,jax_Tanhshrink,jax_Softmax,jax_LogSoftmax
                    )


class Tanh(Tensor):
    def __init__(self, X):
        if isinstance(X, Tensor):
            array = jnp.array(X.data)
            data = jnp.tanh(array)
            super().__init__(data, _parents=[X], _ops="Tanh", requires_grad=X.requires_grad)
        else:
            raise TypeError("Expected a Tensor as input")

    def _backward(self,grad):
        super()._backward(grad)


class ReLU(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_ReLU(array)
            super().__init__(data, _parents=[X],_ops = "ReLU")
        else:
            raise TypeError("Expected a Tensor as input")
    def _backward(self,grad):
        super()._backward(grad)

class ELU(Tensor):
    def __init__(self,X,alpha = 1.0):
        """
        alpha (float): the value for the ELU formulation. Default: 1.0
        """
        if isinstance(X,Tensor):
            self.alpha = alpha
            array = jnp.array(X.data)
            data = jax_ELU(array, alpha)
            super().__init__(data, _parents=[X], _ops = "ELU")
        else:
            raise TypeError("Expected a Tensor as input")
    def _backward(self,grad):
        super()._backward(grad)

class HardShrink(Tensor):
    def __init__(self,X,lambd = 0.5):
        """
        lambd (float): the value for the Hardshrink formulation. Default: 0.5
        """
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            self.lambd = lambd
            data = jax_HardShrink(array,lambd)
            super().__init__(data, _parents=[X],_ops = "HardShrink")
        else:
            raise TypeError("Expected a Tensor as input")
        
    def _backward(self,grad):
        super()._backward(grad)

class Hardsigmoid(Tensor):
    def __init__(self, X):
        if isinstance(X, Tensor):
            array = jnp.array(X.data)
            data = jax_Hardsigmoid(array)
            super().__init__(data, _parents=[X],_ops = "Hardsigmoid")
        else:
            raise TypeError("Expected a Tensor as input")
        
    def _backward(self,grad):
        super()._backward(grad)

    
class Hardtanh(Tensor):
    def __init__(self, X, min_val=-1.0, max_val=1.0):
        """
        Args:
        min_val (float): minimum value of the linear region range. Default: -1
        max_val (float): maximum value of the linear region range. Default: 1
        
        """
        if isinstance(X, Tensor):
            self.max_val = max_val
            self.min_val = min_val
            array = jnp.array(X.data)
            data = jax_Hardtanh(array, min_val, max_val)
            super().__init__(data, _parents=[X],_ops = "Hardtanh")
        else:
            raise TypeError("Expected a Tensor as input")
        
    def _backward(self,grad):
        super()._backward(grad)

class Hardswish(Tensor):    
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_Hardswish(array)
            super().__init__(data,_parents = [X],_ops = "Hardswish")
        else:
            raise TypeError("Expected a Tensor as input")
        
    def _backward(self,grad):
        super()._backward(grad)

class LeakyReLU(Tensor):
    def __init__(self,X,negative_slope = 1e-2):
        """
        Args:
        negative_slope (float): Controls the angle of the negative slope (which is used for negative input values). Default: 1e-2
        """
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            self.neg_slope = negative_slope
            data = jax_LeakyReLU(array, negative_slope)
            super().__init__(data, _parents=[X],_ops = "LeakyReLU")
        else:
            raise TypeError("Expected a Tensor as input")    
        
    def _backward(self,grad):
        super()._backward(grad)

class LogSigmoid(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_LogSigmoid(array)
            super().__init__(data, _parents=[X],_ops = "LogSigmoid")
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self,grad):
        super()._backward(grad)

class ReLU6(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_ReLU6(array)
            super().__init__(data, _parents=[X],_ops = "ReLU6")
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self,grad):
        super()._backward(grad)


class Sigmoid(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_Sigmoid(array)
            super().__init__(data, _parents=[X],_ops = "Sigmoid")
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self,grad):
        super()._backward(grad)


class Softshrink(Tensor):
    def __init__(self,X,lambd = 0.5):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            self.lambd = lambd
            data = jax_Softshrink(array,lambd)
            super().__init__(data, _parents=[X],_ops = "Softshrink")
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self,grad):
        super()._backward(grad)

class Softsign(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_Softsign(array)
            super().__init__(data, _parents=[X],_ops = "Softsign")
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self,grad):
        super()._backward(grad)

class Tanhshrink(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_Tanhshrink(array)
            super().__init__(data, _parents=[X],_ops = "Tanhshrink")
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self,grad):
        super()._backward(grad)


class Softmax(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_Softmax(array)
            super().__init__(data, _parents=[X],_ops = "Softmax")
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self,grad):
        super()._backward(grad)


class LogSoftmax(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_LogSoftmax(array)
            super().__init__(data, _parents=[X], _ops="LogSoftmax")
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self, grad):
        super()._backward(grad)