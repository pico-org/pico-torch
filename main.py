import pico_torch as pt
A = pt.Tensor([1, 2, -9])
B = pt.Tensor([2, 6, 9])
C = pt.ReLU(A) 
print(C)