""" Neural network activation functions and their derivatives
"""
import numpy as np


def activation(name, stimulus):
    """returns result of activation function given name and input
    """
    return {
        "vacuum":vacuum(stimulus),
        "linear":linear(stimulus),
        "logistic":logistic(stimulus),
        "relu":relu(stimulus),
        "bentiden":bentiden(stimulus),
        "gaussian":gaussian(stimulus),
        "softmax":softmax(stimulus)
    }.get(name, vacuum(stimulus))

def dactivation(name, stimulus):
    """returns result of named activation function derivative with respect to
    stimulus
    """
    return {
        "vacuum":vacuum(stimulus),
        "linear":dlinear(stimulus),
        "logistic":dlogistic(stimulus),
        "relu":drelu(stimulus),
        "bentiden":dbentiden(stimulus),
        "gaussian":dgaussian(stimulus),
        "softmax":dsoftmax(stimulus)
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

#----------------------------------------------------

def relu(stimulus):
    """definition of leaky relu function
    """
    return np.where(stimulus > 0, stimulus, .01*stimulus)

def drelu(stimulus):
    """definition of deriv of leaky relu function with respect to stimulus
    """
    return np.where(stimulus > 0, 1, .01)
#----------------------------------------------------

def bentiden(stimulus):
    """definition of bent identity function
    """
    farray = np.multiply(stimulus, stimulus)
    farray = np.add(farray, 1)
    farray = np.sqrt(farray)
    farray = np.add(farray, -1)
    farray = np.divide(farray, 2)
    farray = np.add(farray, stimulus)
    return farray

def dbentiden(stimulus):
    """definition of deriv of bent idenity function with respect to stimulus
    """
    farray = np.multiply(stimulus, stimulus)
    farray = np.add(farray, 1)
    farray = np.sqrt(farray)
    farray = np.multiply(2, farray)
    farray = np.divide(stimulus, farray)
    farray = np.add(farray, 1)
    return farray

#----------------------------------------------------

def gaussian(stimulus):
    """definition of gaussian function
    """
    return np.exp(-np.multiply(stimulus, stimulus))

def dgaussian(stimulus):
    """definition of deriv of gaussian function with respect to stimulus
    """
    gauss = gaussian(stimulus)
    return np.multiply(-2.0, np.multiply(stimulus, gauss))

#----------------------------------------------------

def softmax(stimulus):
    """definition of softmax function
    """
    exps = np.exp(stimulus - np.max(stimulus))
    return exps / np.sum(exps)

def dsoftmax(stimulus):
    """definition of deriv of softmax function with respect to stimulus
    """
    #dp_i/da_j = p_i(krondelta(i,j)-p_j)
    return np.zeros(stimulus.shape)

#----------------------------------------------------


#step
#tanh
#arctan
#prelu
#elu
#softplus
