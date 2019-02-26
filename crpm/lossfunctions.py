""" Neural network loss functions and their derivatives with respect to prediciton
"""
import numpy as np

def loss(name, pred, target):
    """returns result of loss function given name and predictions and targets
    """
    if name == "bce":
        return bce(pred, target)
    if name == "mse":
        return mse(pred, target)
    #default return function
    return mse(pred, target)
#----------------------------------------------------

def bce(pred, target):
    """Binary cross entropy error function
    """
    pred1 = np.subtract(1, pred)
    logpred = np.log(pred)
    logpred1 = np.log(pred1)
    dloss = np.where(target == 1, np.divide(-1, pred), np.divide(1, pred1))
    cost = np.where(target == 1, logpred, logpred1).mean()
    return -cost, dloss

#----------------------------------------------------

def mse(pred, target):
    """Mean squared error function
    """
    dloss = np.subtract(pred, target)
    cost = np.square(dloss).mean()/2.0
    return cost, dloss

#----------------------------------------------------
