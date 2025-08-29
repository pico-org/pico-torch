import jax # type: ignore
import jax.numpy as jnp  # type: ignore
import random

class Tensor:
    def __init__(self, data, dtype=None, requires_grad=False, _parents=None,_ops = ""):
        self.data = jnp.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self._ops = _ops

        self._grad = jnp.zeros(self.data.shape) if self.requires_grad else None
        self._grad_fn = None
        self._parents = [] if _parents is None else _parents
        self.size = None

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        if isinstance(other, (int, float)):
            data = self.data + other
            return Tensor(data, _parents=[self], _ops="add_scalar", requires_grad=self.requires_grad)
        else:
            data = self.data + other.data
            return Tensor(data, _parents=[self, other], _ops="+", requires_grad=(self.requires_grad or other.requires_grad))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            data = self.data - other
            return Tensor(data, _parents=[self], _ops="sub_scalar", requires_grad=self.requires_grad)
        else:
            data = self.data - other.data
            return Tensor(data, _parents=[self, other], _ops="-", requires_grad=(self.requires_grad or other.requires_grad))

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            data = other - self.data
            return Tensor(data, _parents=[self], _ops="rsub_scalar", requires_grad=self.requires_grad)
        else:
            data = other.data - self.data
            return Tensor(data, _parents=[other, self], _ops="-", requires_grad=(self.requires_grad or other.requires_grad))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            data = self.data * other
            result = Tensor(data, _parents=[self], _ops="mul_scalar", requires_grad=self.requires_grad)
            result._scalar = other # type: ignore
            return result
        else:
            data = self.data * other.data
            return Tensor(data, _parents=[self, other], _ops="*", requires_grad=(self.requires_grad or other.requires_grad))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            data = self.data / other
            result = Tensor(data, _parents=[self], _ops="div_scalar", requires_grad=self.requires_grad)
            result._scalar = other # type: ignore
            return result
        else:
            data = self.data / other.data
            return Tensor(data, _parents=[self, other], _ops="/", requires_grad=(self.requires_grad or other.requires_grad))

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            data = other / self.data
            result = Tensor(data, _parents=[self], _ops="rdiv_scalar", requires_grad=self.requires_grad)
            result._scalar = other # type: ignore
            return result
        else:
            return other / self

    def __neg__(self):
        data = -self.data
        return Tensor(data, _parents=[self], _ops="neg", requires_grad=self.requires_grad)

    def __pow__(self, num):
        if isinstance(num, (int, float)):
            data = self.data**num
            result = Tensor(data, _parents=[self], _ops="pow", requires_grad=self.requires_grad)
            result._power = num # type: ignore
            return result
        else:
            raise TypeError(f"Power operation not supported for type {type(num)}")

    def __eq__(self, other):
        if isinstance(other, Tensor):
            data = self.data == other.data
            return Tensor(data, _parents=[self, other])
        else:
            data = self.data == other
            return Tensor(data, _parents=[self])

    def __ne__(self, other):
        if isinstance(other, Tensor):
            data = self.data != other.data
            return Tensor(data, _parents=[self, other])
        else:
            data = self.data != other
            return Tensor(data, _parents=[self])
    
    def __matmul__(self,other):
        if isinstance(other,Tensor):
            x = self.data
            y = other.data
            return Tensor(jnp.matmul(x,y), _parents=[self,other], _ops="matmul", requires_grad=(self.requires_grad or other.requires_grad))
        else:
            raise ValueError("Need Tensor obj for matmul")

    def shape(self):
        return self.data.shape

    def reshape(self,size):
        data = self.data.reshape(size)
        return Tensor(data)
    
    def unsqueeze(self, dim):
        data = jnp.expand_dims(self.data, axis=dim)
        return Tensor(data, requires_grad=self.requires_grad, _parents=[self], _ops="unsqueeze")
    
    def squeeze(self, dim=None):
        data = jnp.squeeze(self.data,axis=dim)
        return Tensor(data, requires_grad=self.requires_grad, _parents=[self], _ops="squeeze")
    
    def permute(self,perm):
        data = jnp.permute_dims(self.data,axes=perm)
        return Tensor(data)

    def flatten(self):
        _shape = self.data.shape
        total_element = 1
        for i in _shape:
            total_element*=i
        data = self.data.reshape((1,total_element))
        return Tensor(data)
    
    def sum(self,dim):
        data = jnp.sum(self.data,dim)
        return Tensor(data)
        
    def argmax(self, dim=None):
        return jnp.argmax(self.data, axis=dim)
    
    def argmin(self, dim=None):
        return jnp.argmin(self.data,axis=dim)
    

    def _backward(self, grad=0.3):
        """Backward pass for gradient computation. Implementation in _backward.py"""
        pass  # This will be overridden by the decorator in _backward.py


class empty(Tensor):
    def __new__(cls,shape:tuple):
        _size = 1
        a = Tensor(jnp.empty(shape),requires_grad=True)
        for i in list(shape):
            _size*=i
        size = _size
        return a        
        

class Random(Tensor):
    def __new__(cls, shape: tuple):
        _data = [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]  
        for i in range(shape[0]):
            for j in range(shape[1]):
                _data[i][j] = random.random()
        return Tensor(_data)



