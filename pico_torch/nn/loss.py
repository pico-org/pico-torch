import jax 
import jax.numpy as jnp
from ..tensor.ops import Tensor
from ._loss_utils import (jax_l1loss,jax_mse,jax_cel,jax_cel_indices,jax_hl,jax_SL1l,jax_SML)

class L1loss:
    def __init__(self):
        pass
    def __call__(self,prediction,ground_truth):
        if isinstance(prediction,Tensor):
            self.p = jnp.array(prediction.data)
        else:
            print("given prediction is not in correct format")
            
        if isinstance(ground_truth,Tensor):
            self.gt = jnp.array(ground_truth.data)
        else:
            self.gt = jnp.array(ground_truth)
            
        if self.p.shape != self.gt.shape:
            return RuntimeError("wrong shape provided")
        else:
            self.num_element = self.p.size
            return Tensor(jax_l1loss(self.p,self.gt,self.num_element),_parents = [prediction,ground_truth])
    
class MSELoss:
    def __init__(self):
        pass
    def __call__(self,prediction,ground_truth):
        if isinstance(prediction,Tensor):
            self.p = jnp.array(prediction.data)
        else:
            print("given prediction is not in correct format")
        
        if isinstance(ground_truth,Tensor):
            self.gt = jnp.array(ground_truth.data)
        else:
            self.gt = jnp.array(ground_truth)

        if self.p.shape != self.gt.shape:
            return RuntimeError("wrong shape provided")
        else:
            self.num_element = self.p.size
            return Tensor(jax_mse(self.p,self.gt,self.num_element),_parents = [prediction,ground_truth])
                
class CrossEntropyLoss:
    def __init__(self, from_logits=True):
        """
        Args:
            from_logits (bool): If True, expects raw logits. If False, expects probabilities.
        """
        self.from_logits = from_logits
        
    def __call__(self, prediction, ground_truth):
        if isinstance(prediction, Tensor):
            self.p = jnp.array(prediction.data)
        else:
            print("given prediction is not in correct format")
            return None
        
        if isinstance(ground_truth, Tensor):
            self.gt = jnp.array(ground_truth.data)
        else:
            self.gt = jnp.array(ground_truth)

        self.num_element = self.gt.shape[0] if self.gt.ndim > 0 else 1

        if self.gt.ndim == 1 and self.gt.dtype in [jnp.int32, jnp.int64]:
            return jax_cel_indices(self.p, self.gt.astype(jnp.int32), self.num_element)
        else:
            return Tensor(jax_cel(self.p, self.gt, self.num_element),_parents = [prediction,ground_truth])
        

class HuberLoss:
    def __init__(self,delta = 1.0):
        self.delta = delta
    
    def __call__(self,prediction,ground_truth):
        if isinstance(prediction,Tensor):
            self.p = jnp.array(prediction.data)
        else:
            print("given prediction is not in correct format")
            return None

        if isinstance(ground_truth,Tensor):
            self.gt = jnp.array(ground_truth.data)
        else:
            self.gt = jnp.array(ground_truth)
        
        if self.p.shape != self.gt.shape:
            return RuntimeError("wrong shape provided")
        else:
            return Tensor(jax_hl(self.p,self.gt,self.delta),_parents = [prediction,ground_truth])
        

class SmoothL1Loss:
    def __init__(self,beta = 1.0):
        self.beta = beta
    
    def __call__(self,prediction,ground_truth):
        if isinstance(prediction,Tensor):
            self.p = jnp.array(prediction.data)
        else:
            print("given prediction is not in correct format")
            return None

        if isinstance(ground_truth,Tensor):
            self.gt = jnp.array(ground_truth.data)
        else:
            self.gt = jnp.array(ground_truth)
        
        if self.p.shape != self.gt.shape:
            return RuntimeError("wrong shape provided")
        else:
            return Tensor(jax_SL1l(self.p,self.gt,self.beta),_parents = [prediction,ground_truth])
        

class SoftMarginLoss:
    def __init__(self):
        pass

    def __call__(self,prediction,ground_truth):
        if isinstance(prediction,Tensor):
            self.p = jnp.array(prediction.data)
        else:
            print("given prediction is not in correct format")
            return None

        if isinstance(ground_truth,Tensor):
            self.gt = jnp.array(ground_truth.data)
        else:
            self.gt = jnp.array(ground_truth)
        
        if self.p.shape != self.gt.shape:
            return RuntimeError("wrong shape provided")
        else:
            return Tensor(jax_SML(self.p,self.gt),_parents = [prediction,ground_truth])