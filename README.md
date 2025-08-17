# pico-torch
A educational purpose implementation of lightweight deeplearning library in python and jax.

## tensor:
it has own tensor data-structure with arithmatic operation support.

### basic operations:
- addition (+)
- subtraction (-) 
- multiplication (*)
- division (/)
- power (**)
- negative (-)

### how to use:
```python
import pico_torch as pt

# create tensors
A = pt.Tensor([1,2,3], requires_grad=True)
B = pt.Tensor([0.5,2.6,3.8], requires_grad=True)

# arithmetic operations
C = A + B
D = A - B
E = A * B
F = A / B
G = A ** 2
H = -A

# compute gradients
C._backward()
print(A._grad)
```

## activation functions:
- ReLU
- Tanh
- Sigmoid
- ELU
- HardShrink
- Hardsigmoid
- Hardtanh
- Hardswish
- LeakyReLU
- LogSigmoid
- ReLU6
- Softshrink
- Softsign
- Tanhshrink
- Softmax
- LogSoftmax

### how to use:
```python
import pico_torch as pt

A = pt.Tensor([1,2,3], requires_grad=True)
relu = pt.ReLU(A)
tanh = pt.Tanh(A)
sigmoid = pt.Sigmoid(A)

relu._backward()
print(A._grad)
```

## autograd:
automatic differentiation is supported for basic operations and activation functions.
gradients are computed when calling _backward() method.

## requirements:
- python 3.x
- jax
- jax.numpy
