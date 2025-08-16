from .tensor import Tensor
import jax.numpy as jnp  # type: ignore
from jax import jit  # type: ignore

from .utils import (jax_tanh,jax_ReLU, jax_ELU, jax_HardShrink, jax_Hardsigmoid,jax_Hardtanh,
                    jax_Hardswish,jax_LeakyReLU,jax_LogSigmoid,jax_ReLU6,jax_Sigmoid,jax_Softshrink,
                    jax_Softsign,jax_Tanhshrink,jax_Softmax,jax_LogSoftmax
                    )

__all__ = ["Tensor"]

# Warmup JIT for scalar and typical tensor shape
jax_tanh(2)  # scalar warmup
jax_tanh(jnp.ones((3,)))  # vector warmup 

import builtins
builtins.Tensor = Tensor

class Tanh(Tensor):
    def __init__(self, X):
        if isinstance(X, Tensor):
            array = jnp.array(X.data)
            data = jnp.tanh(array)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input")

    def _backward(self):
        pass


class ReLU(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_ReLU(array)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input")
    def _backward(self):
        pass

class ELU(Tensor):
    def __init__(self,X,alpha = 1.0):
        """
        alpha (float): the value for the ELU formulation. Default: 1.0
        """
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_ELU(array, alpha)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input")
    def _backward(self):
        pass

class HardShrink(Tensor):
    def __init__(self,X,lambd = 0.5):
        """
        lambd (float): the value for the Hardshrink formulation. Default: 0.5
        """
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_HardShrink(array,lambd)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input")
        
    def _backward(self):
        pass

class Hardsigmoid(Tensor):
    def Hardsigmoid(self,X):

        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_Hardsigmoid(array)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input")
        
    def _backward(self):
        pass

    
class Hardtanh(Tensor):
    def Hardtanh(self,X,min_val=-1.0, max_val=1.0):
        """
        Args:
        min_val (float): minimum value of the linear region range. Default: -1
        max_val (float): maximum value of the linear region range. Default: 1
        
        """
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_Hardtanh(array)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input")
        
    def _backward(self):
        pass

class Hardswish(Tensor):    
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_Hardswish(array)
            super().__init__(data,_parents = [X])
        else:
            raise TypeError("Expected a Tensor as input")
        
    def _backward(self):
        pass

class LeakyReLU(Tensor):
    def __init__(self,X,negative_slope = 1e-2):
        """
        Args:
        negative_slope (float): Controls the angle of the negative slope (which is used for negative input values). Default: 1e-2
        """
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_LeakyReLU(array)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input")    
        
    def _backward(self):
        pass

class LogSigmoid(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_LogSigmoid(array)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self):
        pass

class ReLU6(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_ReLU6(array)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self):
        pass

class Sigmoid(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_Sigmoid(array)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self):
        pass


class Softshrink(Tensor):
    def __init__(self,X,lmabd = 0.5):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_Softshrink(array)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self):
        pass

class Softsign(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_Softsign(array)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self):
        pass

class Tanhshrink(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_Tanhshrink(array)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self):
        pass


class Softmax(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_Softmax(array)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self):
        pass


class LogSoftmax(Tensor):
    def __init__(self,X):
        if isinstance(X,Tensor):
            array = jnp.array(X.data)
            data = jax_LogSoftmax(array)
            super().__init__(data, _parents=[X])
        else:
            raise TypeError("Expected a Tensor as input") 
    def _backward(self):
        pass