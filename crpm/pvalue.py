""" Calcualte p-values, ROC, AUC, and proportion of significant observations for
a set of observations given the null hypothesis distribution

    Args:
        variable: array of observed values
        hypothesis: optional null hypotheis distribution (beta distribution by default)
        alpha: optional significance parameter (.05 by default)
    Returns:
        pvalues: for every observation in variable
        ROC: on a grid of 1000 points
        AUC: integral of ROC
        proportion of significant observations
"""

import numpy as np

def pvalue(variable=None, hypothesis=None, alpha=.05):
    """ calcualte pvalues, AUC and fraction of significant observations
    """
    #set model
    if variable is None:
        variable = np.random.beta(a=3, b=5, size=5000)

    else:
        variable = np.array(variable)

    #set null-hypothesis
    if hypothesis is None:
        hypothesis = np.random.beta(a=5, b=5, size=1000)
    else:
        hypothesis = np.array(hypothesis)

    #calculate prob of left-tail event p(H<=x|H) for every instance of X
    prob = []
    for var in variable:
        prob.append((hypothesis <= var).sum())
    #normalize p
    prob = np.divide(prob, hypothesis.size)

    #scan alpha from 0 to 1 and find prob(p<=alpha)
    scanprob = []
    alphagrid = np.linspace(0, 1, num=1000)
    for val in alphagrid:
        #calculate prob p<=alpha
        scanprob.append((prob <= val).sum() / variable.size)

    return prob, scanprob, np.sum(prob) / alphagrid.size, (prob <= alpha).sum() /variable.size

def lefttailpvalue(variable=None, hypothesis=None):
    """ calcualte left-tail pvalues
    """
    #set model
    if variable is None:
        variable = np.random.beta(a=3, b=5, size=5000)

    else:
        variable = np.array(variable)

    #set null-hypothesis
    if hypothesis is None:
        hypothesis = np.random.beta(a=5, b=5, size=1000)
    else:
        hypothesis = np.array(hypothesis)

    #calculate prob of left-tail event p(H<=x|H) for every instance of X
    prob = []
    for var in variable:
        prob.append((hypothesis <= var).sum())
    #normalize p
    prob = np.divide(prob, hypothesis.size)

    return prob

def righttailpvalue(variable=None, hypothesis=None):
    """ calcualte left-tail pvalues
    """
    #set model
    if variable is None:
        variable = np.random.beta(a=3, b=5, size=5000)

    else:
        variable = np.array(variable)

    #set null-hypothesis
    if hypothesis is None:
        hypothesis = np.random.beta(a=5, b=5, size=1000)
    else:
        hypothesis = np.array(hypothesis)

    #calculate prob of right-tail event p(H>=x|H) for every instance of X
    prob = []
    for var in variable:
        prob.append((hypothesis >= var).sum())
    #normalize p
    prob = np.divide(prob, hypothesis.size)

    return prob
