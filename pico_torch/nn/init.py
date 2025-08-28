import jax 
import jax.numpy as jnp
import time
from ..tensor.ops import Tensor

_seed_state = int(time.time() * 1000000) % (2**32)

def calculate_gain(nonlinearity, **kwargs):
    list_ = ["Conv1D", "Conv2D", "Conv3D", "Linear", "Sigmoid"]
    
    if nonlinearity in list_:
        return 1
    elif nonlinearity == "Tanh":
        return 5/3
    elif nonlinearity == "ReLU":
        return 2**0.5
    elif nonlinearity == "Leaky Relu":
        negative_slope = kwargs.get("negative_slope", 0.01) 
        return (2.0 / (1 + negative_slope**2))**0.5
    elif nonlinearity == "SELU":
        return 3/4
    

def _xorshift32():
    global _seed_state
    _seed_state ^= _seed_state << 13
    _seed_state ^= _seed_state >> 17
    _seed_state ^= _seed_state << 5
    _seed_state = _seed_state % (2**32) 
    return _seed_state

def _uniform_scratch(shape):
    flat_size = int(jnp.prod(jnp.array(shape)))
    random_floats = []
    for _ in range(flat_size):
        rand_int = _xorshift32()
        rand_float = float(rand_int) / (2**32)
        random_floats.append(rand_float)
    return jnp.array(random_floats, dtype=jnp.float32).reshape(shape)

def uniform_(t, a=0.0, b=1.0):
    shape = t.shape()
    uniform_data = _uniform_scratch(shape)
    t.data = a + (b - a) * uniform_data


def _box_muller_scratch(shape):
    flat_size = int(jnp.prod(jnp.array(shape)))
    pairs_needed = (flat_size + 1) // 2
    
    normal_numbers = []
    for _ in range(pairs_needed):
        u1 = _xorshift32() / (2**32)
        u2 = _xorshift32() / (2**32)
        
        u1 = max(u1, 1e-10)
        z0 = jnp.sqrt(-2.0 * jnp.log(u1)) * jnp.cos(2.0 * jnp.pi * u2)
        z1 = jnp.sqrt(-2.0 * jnp.log(u1)) * jnp.sin(2.0 * jnp.pi * u2)
        normal_numbers.extend([z0, z1])
    normal_array = jnp.array(normal_numbers[:flat_size], dtype=jnp.float32)
    return normal_array.reshape(shape)



def normal_(t, mean=0.0, std=1.0):
    shape = t.shape()
    normal_data = _box_muller_scratch(shape)
    t.data = normal_data * std + mean


def constant_(t,value):
    t.data+=value


def ones_(t):
    t.data+=1

def zeros_(t):
    pass

def eye_(t):
    rows, cols = t.shape()
    r = jnp.arange(rows)[:, None] 
    c = jnp.arange(cols)[None, :]  
    t.data = (r == c).astype(t.data.dtype)

    
