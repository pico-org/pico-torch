import pico_torch as pt
A = pt.Tensor([-4, 0.3, 9])
B = pt.Tensor([2, 6, 9])
C = pt.LogSoftmax(A) 
# print(C._parents)
# print(type(C))
# C._backward()
print(C)

