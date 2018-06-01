""" Neural network activation functions and their derivatives
"""
import numpy as np


def activation(name, stimulus):
    """returns result of activation function given name and input
    """
    return {
        "vacuum":vacuum(stimulus),
        "linear":linear(stimulus),
        "logistic":logistic(stimulus)
    }.get(name, vacuum(stimulus))

def dactivation(name, stimulus):
    """returns result of named activation function derivative with respect to
    stimulus
    """
    return {
        "vacuum":vacuum(stimulus),
        "linear":dlinear(stimulus),
        "logistic":dlogistic(stimulus)
    }.get(name, vacuum(stimulus))

#----------------------------------------------------

def vacuum(stimulus):
    """Vacuum function converts input into zeros
    """
    return np.zeros(stimulus.shape)

#----------------------------------------------------

def linear(stimulus):
    """definition of linear function
    """
    return stimulus

def dlinear(stimulus):
    """definition of deriv of linear function with respect to stimulus
    """
    return np.ones(stimulus.shape)

#----------------------------------------------------

def logistic(stimulus):
    """definition of logistic function
    """
    return 1/(1 + np.exp(-stimulus))

def dlogistic(stimulus):
    """definition of deriv of logistic function with respect to stimulus
    """
    logi = logistic(stimulus)
    return np.multiply(logi, 1-logi)

#step
#softmax
#tanh
#arctan
#relu
#prelu
#elu
#softplus
#gaussian
