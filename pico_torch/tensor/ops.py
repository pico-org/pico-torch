import jax # type: ignore
import jax.numpy as jnp  # type: ignore


class Tensor:
    def __init__(self, data, dtype=None, requires_grad=False, _parents=None,_ops = ""):
        self.data = jnp.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self._ops = _ops

        self._grad = jnp.zeros(self.data.shape) if self.requires_grad is not None else None
        self._grad_fn = None
        self._parents = [] if _parents is None else _parents

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        if isinstance(other, (int, float)):
            data = self.data + other
            return Tensor(data, _parents=[self], _ops="add_scalar")
        else:
            data = self.data + other.data
            return Tensor(data, _parents=[self, other],_ops = "+")

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            data = self.data - other
            return Tensor(data, _parents=[self],_ops = "sub_scalar")
        else:
            data = self.data - other.data
            return Tensor(data, _parents=[self, other], _ops = "-")

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            data = other - self.data
            return Tensor(data, _parents=[self], _ops = "rsub_scalar")
        else:
            data = other.data - self.data
            return Tensor(data, _parents=[other, self], _ops = "-")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            data = self.data * other
            result = Tensor(data, _parents=[self], _ops="mul_scalar")
            result._scalar = other
            return result
        else:
            data = self.data * other.data
            return Tensor(data, _parents=[self, other],_ops = "*")

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            data = self.data / other
            result = Tensor(data, _parents=[self],_ops = "/")
            result._scalar = other
            return result
        else:
            data = self.data / other.data
            return Tensor(data, _parents=[self, other],_ops = "/")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            data = other / self.data
            result = Tensor(data, _parents=[self],_ops = "rdiv_scalar")
            result._scalar = other
            return result
        else:
            return other / self

    def __neg__(self):
        data = -self.data
        return Tensor(data, _parents=[self], _ops = "neg")

    def __pow__(self, num):
        if isinstance(num, (int, float)):
            data = self.data**num
            result = Tensor(data, _parents=[self], _ops = "pow")
            result._power = num
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
