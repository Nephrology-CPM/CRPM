""" Neural network cost functions and their derivatives with respect to prediciton
"""
import numpy as np

def loss(name, pred, target):
    """returns result of loss function given name and predictions and targets
    """
    return {
        "mse":mse(pred, target),
    }.get(name, mse(pred, target))

#----------------------------------------------------

def mse(pred, target):
    """Mean squared error function
    """
    cost = np.square(np.subtract(pred, target)).mean()/2.0
    dloss = np.subtract(pred, target)
    return cost, dloss

#----------------------------------------------------


#MSE
#crossentropy
