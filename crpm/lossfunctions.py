""" Neural network loss functions and their derivatives with respect to prediciton
"""
import numpy as np

def loss(name, pred, target):
    """returns result of loss function given name and predictions and targets
    """
    return {
        "mse":mse(pred, target),
        #"bce":bce(pred, target),
    }.get(name, mse(pred, target))

#----------------------------------------------------

def mse(pred, target):
    """Mean squared error function
    """
    dloss = np.subtract(pred, target)
    cost = np.square(dloss).mean()/2.0
    return cost, dloss

#----------------------------------------------------

def bce(pred, target):
    """Binary cross entropy error function
    """
    pred1 = np.subtract(1, pred)
    targ1 = np.subtract(1, target)
    dloss = np.add(np.divide(targ1, pred1), np.divide(target, pred))
    cost = np.add(np.multiply(target, np.log(pred)),
                  np.multiply(targ1, np.log(pred1))).mean()
    return -cost, -dloss

#----------------------------------------------------
