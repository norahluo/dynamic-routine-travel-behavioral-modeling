#!/usr/bin/env python
# coding: utf-8


import numpy as np

def sgd(x, dx, config=None):
    """
    Performs vanilla stochastic gradient descent.
    """
    if config is None:
        config = {}
        
    config.setdefault('learning_rate', 1e-2)
    
    next_x = None
    
    alpha = config['learning_rate']
    next_x = x - alpha * dx
    
    return next_x, config

def sgd_momentum(x, dx, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a moving
      average of the gradients.
    """
    if config is None: 
        config = {}
        
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    config.setdefault('velocity', np.zeros_like(x))

    next_x = None
    
    m = config['momentum']
    alpha = config['learning_rate']    
    config['velocity'] = m*config['velocity'] + dx
    next_x = x - alpha * config['velocity']

    return next_x, config


def rmsprop(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared gradient
    values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: 
        config = {}
        
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None

    alpha = config['learning_rate']
    beta = config['decay_rate']
    epsilon = config['epsilon']
    config['cache'] = beta * config['cache'] + (1-beta) * dx**2
    next_x = x - alpha/(config['cache']+epsilon)**0.5 * dx

    return next_x, config

def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the gradient 
    and its square and a bias correction term.
    """
    if config is None:
        config = {}
        
    # Set default values for configuration parameters    
    config.setdefault('learning_rate', 1e-1)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)
    config.setdefault('verbose', False)

    # Extract configuration parameters
    alpha = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    epsilon = config['epsilon']
    verbose = config['verbose']

    # Increment time step
    config['t'] += 1

    # Update biased first moment estimate
    config['m'] = beta1 * config['m'] + (1-beta1) * dx 
    # Update biased second raw moment estimate
    config['v'] = beta2 * config['v'] + (1-beta2) * (dx**2) 

    # Compute biased-corrected first and second moments
    m_hat = config['m']/(1-beta1**config['t'])
    v_hat = config['v']/(1-beta2**config['t'])
    
    # Compute adaptive gradient
    g = m_hat / (np.sqrt(v_hat) + epsilon)

    # Update parameters
    next_x = x - alpha * g

    # Optional debugging output
    if verbose:
        print('Step {}'.format(config['t']))
        print('dx is {:3f}, m is {:.3f}, m_hat is {:.3f}, v is {:.3f}, v_hat is {:.3f}, g is {:.3f}.'.format(dx, config['m'], config['v'], m_hat, v_hat, g))
        print('x is {:3f}, dx is {:3f}, g is {:3f}, next_x is {:3f}.'.format(x, dx, g, next_x))
        
    return next_x, config

