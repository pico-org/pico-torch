import pico_torch as pt
import jax.numpy as jnp
a = [[1,2,3],[3,4,5]]
b = [[0.5,2.6,3.8],[3,4,5]]
A = pt.Tensor(a,requires_grad=True)
B = pt.Tensor(b,requires_grad=True)
C = A - B
C.requires_grad = True
C._grad = jnp.ones_like(C.data)
C._backward(grad = C._grad)

D = pt.ReLU(C)
D._backward()
print(type(D))
# print(C._parents)
# print(type(C))
# C._backward()
print(type(A))
print(A._grad)
