""" Neural network loss functions and their derivatives with respect to prediciton
"""
import numpy as np

def loss(name, pred, target, logit=None):
    """returns result of loss function given name and predictions and targets
    """
    if name == "bce":
        return bce(pred, target, logit)
    if name == "mse":
        return mse(pred, target)
    if name == "upmse":
        return upmse(pred, target)
    if name == "iden":
        return 0, np.ones(pred.shape)
    #default return function
    return mse(pred, target)
#----------------------------------------------------

def bce(pred, target, logit=None):
    """Binary cross entropy error function
        Will calcualate bce from top layer logit directly when logit is provided.
        Assumes logit provided is to a logisitc top layer.
    """
    if logit is None:
        #limit predictions
        pred = np.where(pred>=1, 1-np.finfo(float).eps, pred)
        pred = np.where(pred<=0, np.finfo(float).eps, pred)

        pred1 = np.subtract(1, pred)
        logpred = np.log(pred)
        logpred1 = np.log(pred1)
        dloss = np.where(target == 1, np.divide(1, pred), np.divide(-1, pred1))
        cost = np.where(target == 1, logpred, logpred1).mean()
    else:
        target1 = np.subtract(1, pred)
        cost = np.where(logit > 0,
                        np.subtract(-np.multiply(target1,logit),
                                    np.log(1+np.exp(-logit)),
                        np.subtract(np.multiply(target,logit),
                                    np.log(1+np.exp(logit))))
        dloss = np.where(logit > 0,
                         np.multiply(np.subtract(target,
                                                 np.divide(target1,np.exp(-logit))),
                                     (1+np.exp(-logit))),
                         np.multiply(np.subtract(np.divide(target,np.exp(logit)),
                                                 target1),
                                    (1+np.exp(logit)))
#        dloss = np.where(target == 1, -np.exp(-logit)-1, np.exp(logit)+1)
#        cost = np.where(logit < 0,
#                        -np.log(1+np.exp(logit))*(target*(logit+1)+(1-target)),
#                        -np.log(1+np.exp(-logit))*(target+(1-target)*(1-logit)))
        cost = cost.mean()
    return -cost, -dloss

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
