import jax.numpy as jnp
from ..tensor.ops import Tensor,Random,empty
import jax.random as jrm
import pico_torch.nn as nn
import math
from ..nn.Module import Module

class Linear(Module):
    def __init__(self,in_feature,out_feature,bias = True,dtype = None):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.bias =  empty((1,out_feature)) if bias == True else None
        nn.uniform_(self.bias,-math.sqrt(in_feature),math.sqrt(in_feature))
        self.dtype = dtype
        self.weights = empty((in_feature,out_feature))
        nn.uniform_(self.weights,-math.sqrt(in_feature),math.sqrt(in_feature))
        self.add_parameters("weights",self.weights)
        self.add_parameters("bias",self.bias) # type: ignore
        
    def __call__(self,x:Tensor):
        b = x@self.weights
        c = b+self.bias if self.bias is not None else b
        return c
    
    def forward(self,x):
        return self.__call__(x)
