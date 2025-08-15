import jax
import jax.numpy as jnp


class Tensor:
    def __init__(self, data, dtype=None, requires_grad=False, _parents=None):
        self.data = jnp.array(data, dtype=dtype)
        self.requires_grad = requires_grad

        self._grad = None
        self._grad_fn = None
        self._parents = [] if _parents is None else _parents

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        if isinstance(other, (int, float)):
            data = self.data + other
            return Tensor(data, _parents=[self])
        else:
            data = self.data + other.data
            return Tensor(data, _parents=[self, other])

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            data = self.data - other
            return Tensor(data, _parents=[self])
        else:
            data = self.data - other.data
            return Tensor(data, _parents=[self, other])

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            data = other - self.data
            return Tensor(data, _parents=[self])
        else:
            return other - self

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            data = self.data * other
            return Tensor(data, _parents=[self])
        else:
            data = self.data * other.data
            return Tensor(data, _parents=[self, other])

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            data = self.data / other
            return Tensor(data, _parents=[self])
        else:
            data = self.data / other.data
            return Tensor(data, _parents=[self, other])

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            data = other / self.data
            return Tensor(data, _parents=[self])
        else:
            return other / self

    def __neg__(self):
        data = -self.data
        return Tensor(data, _parents=[self])

    def __pow__(self, num):
        if isinstance(num, (int, float)):
            data = self.data**num
            return Tensor(data, _parents=[self])
        elif isinstance(num, Tensor):
            data = self.data**num.data
            return Tensor(data, _parents=[self, num])
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
