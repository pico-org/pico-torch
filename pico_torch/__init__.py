from .tensor.ops import Tensor,empty,Random
from .tensor.activation import (Tanh, ReLU, ELU, HardShrink, Hardsigmoid, Hardtanh,
                               Hardswish, LeakyReLU, LogSigmoid, ReLU6, Sigmoid, Softshrink,
                               Softsign, Tanhshrink, Softmax, LogSoftmax)

from .tensor._backward import (_backward)
def initialize():
    from .tensor.activation_utils import _initialize_activation_functions
    from .tensor.backward_utils import _initialize_backward_functions
    
    _initialize_activation_functions()
    _initialize_backward_functions()

__all__ = ["Tensor", "empty","Random", "Tanh", "ReLU", "ELU", "HardShrink", "Hardsigmoid", "Hardtanh",
           "Hardswish", "LeakyReLU", "LogSigmoid", "ReLU6", "Sigmoid", "Softshrink",
           "Softsign", "Tanhshrink", "Softmax", "LogSoftmax","_backward", "initialize"]

