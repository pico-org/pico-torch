import jax
import jax.numpy as jnp
from ..tensor.ops import Tensor


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}


    def parameters(self):
        for p in self._parameters.values():
            if p is not None: 
                yield p
        
        for module in self._modules.values():
            if module is not None:
                for p in module.parameters():
                    yield p

    def add_parameters(self, name, param):
        self._parameters[name] = param
    
    def add_module(self, name, module):
        self._modules[name] = module
    
    def __setattr__(self, name, value):
        if isinstance(value, Module):
            if hasattr(self, '_modules'):
                self._modules[name] = value
            else:
                super().__setattr__(name, value)
                return
        super().__setattr__(name, value)