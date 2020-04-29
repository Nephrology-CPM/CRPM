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
    if name == "upmse":
        return upmse(pred, target)
    if name == "iden":
        return 0, np.ones(pred.shape)
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

def upmse(pred, target):
    """Un-paired Mean squared error function
    """
    #get number of samples in pred and target
    mobv = target.shape[1]
    mhat = pred.shape[1]

    #init closest sample matrix of shape (mobv, mhat)
    #indicates if sample in mhat is closest to mobv
    # so each row is a one-hot vector of size mhat
    sigma = np.zeros((mobv,mhat))

    #Find closest pred for each target by brute force
    for m in range(mobv):
        sqdist = np.sum(np.square(pred - target[:, m:m+1]), axis=0)
        #set one-hot vector in row m
        sigma[m, np.argmin(sqdist)] = 1

    #Normalize closest sample matrix to have unit sum over mobv
    norm = np.sum(sigma, axis=1, keepdims=True)
    norm = np.where(norm>1, norm, 1) #if colsum is zero then norm is 1
    sigmat = np.divide(sigma, norm)  #to avoid divide by zero here

    dloss = np.subtract(pred, target.dot(sigmat))
    cost = np.square(np.subtract(pred.dot(sigma.T), target)).mean()/2.0
    return cost, dloss

#----------------------------------------------------
