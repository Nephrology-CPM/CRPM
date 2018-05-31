""" Neural network activation functions and their derivatives
"""
import numpy as np


def activation(name, stimulus):
    """returns result of activation function given name and input
    """
    return {
        "vacuum":vacuum(stimulus),
        "linear":linear(stimulus),
    }.get(name, vacuum(stimulus))

def dactivation(name, stimulus):
    """returns result of named activation function derivative with respect to
    stimulus
    """
    return {
        "vacuum":vacuum(stimulus),
        "linear":dlinear(stimulus),
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

#step
#logistic
#softmax
#tanh
#arctan
#relu
#prelu
#elu
#softplus
#gaussian
