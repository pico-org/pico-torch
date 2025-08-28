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

@jax.jit
def jax_hl(x, y, delta):
    diff = jnp.abs(x - y)
    loss = jnp.where(diff <= delta, 0.5 * diff**2, delta * diff - 0.5 * delta**2)
    if isinstance(loss, tuple):
        loss = loss[0]
    return jnp.mean(loss)


@jax.jit 
def jax_SL1l(x,y,beta):
    diff = jnp.abs(x - y)
    loss = jnp.where(diff <= beta, (0.5*(x-y)**2)/beta, diff-(0.5*beta))
    return jnp.mean(loss)

@jax.jit
def jax_SML(x,y):
    _nelement = x.size
    return jnp.sum(jnp.log(1+jnp.exp(-y*x))/_nelement)
