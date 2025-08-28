import jax.numpy as jnp
from ..tensor.ops import Tensor,Random
import jax.random as jrm


class Linear:
    def __init__(self,in_feature,out_feature,bias = True,dtype = None):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.bias = Tensor(jnp.zeros((1,out_feature))) if bias == True else None 
        self.dtype = dtype
        self.weights = Random((in_feature,out_feature))

    def __call__(self,x:Tensor):
        b = x@self.weights
        c = b+self.bias if self.bias is not None else b
        return c
    
    def forward(self,x):
        return self.__call__(x)

