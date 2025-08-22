import jax 
import jax.numpy as jnp
import jax.nn as nn


@jax.jit
def jax_l1loss(x,y,ele):
    return jnp.sum(jnp.abs(x - y))/ele


@jax.jit
def jax_mse(x,y,ele):
    return jnp.sum((x-y)**2)/ele


@jax.jit
def jax_cel(x, y, ele):
    """
    x: logits (raw)
    y: true labels (ohe)
    ele: number of samples 
    """
    prob = nn.softmax(x, axis=-1)
    epsilon = 1e-8
    prob = prob + epsilon
    cross_entropy = -jnp.sum(y * jnp.log(prob), axis=-1)

    return jnp.mean(cross_entropy)


@jax.jit  
def jax_cel_indices(x, y, ele):
    """
    x: logits (raw)
    y: true labels
    ele: number of samples 
    """
    prob = nn.softmax(x, axis=-1)
    epsilon = 1e-8
    log_prob = jnp.log(prob + epsilon)
    
    # karypathy trick!!! ;)
    correct_log_probs = log_prob[jnp.arange(len(y)), y]
    
    return -jnp.mean(correct_log_probs)