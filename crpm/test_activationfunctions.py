""" Test activation functions and their derivatives
"""

import numpy as np
from crpm.activationfunctions import activation
from crpm.activationfunctions import dactivation


def test_vacuum():
    """test vacuum function returns zeros in same shape as input
    """
    #set random Seed
    np.random.seed(1500450271)

    stim = np.ones((3, 5))
    assert np.all(activation("vacuum", stim) == 0)
    assert activation("vacuum", stim).shape == (3, 5)
    assert np.all(dactivation("vacuum", stim) == 0)
    assert dactivation("vacuum", stim).shape == (3, 5)

def test_linear():
    """test linear function returns input and derivative returns
    ones in same shape as input
    """
    #set random Seed
    np.random.seed(1500450271)

    stim = np.random.rand(5, 7)
    assert np.all(activation("linear", stim) == stim)
    assert activation("linear", stim).shape == (5, 7)
    assert np.all(dactivation("linear", stim) == 1)
    assert dactivation("linear", stim).shape == (5, 7)

def test_relu():
    """test relu function returns input for positive and .01 of input for
    negative also check derivative returns ones where positive and .01 where
    negative and both have same shape as input
    """
    #set random Seed
    np.random.seed(1500450271)

    stim = np.random.randn(5, 7)
    assert np.all(activation("relu", stim) == np.where(stim > 0, stim, .01*stim))
    assert activation("relu", stim).shape == (5, 7)
    assert np.all(dactivation("relu", stim) == np.where(stim > 0, 1, .01))
    assert dactivation("relu", stim).shape == (5, 7)

def test_logistic():
    """test logistic function
    """
    #set random Seed
    np.random.seed(1500450271)

    stim = np.random.randn(5, 7)
    #test for correct shape
    assert activation("logistic", stim).shape == (5, 7)
    assert dactivation("logistic", stim).shape == (5, 7)
    #test for numerical stability
    stim *= 1E-3
    assert  not np.any(np.isnan(activation("logistic", stim)))
