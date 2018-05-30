""" Test activation functions and their derivatives
"""

import numpy as np
from crpm.activationfunctions import activation
from crpm.activationfunctions import dactivation

def test_vacuum():
    """test vacuum function returns zeros in same shape as input
    """
    stim = np.ones((3, 5))
    assert np.all(activation("vacuum", stim) == 0)
    assert activation("vacuum", stim).shape == (3, 5)
    assert np.all(dactivation("vacuum", stim) == 0)
    assert dactivation("vacuum", stim).shape == (3, 5)

def test_linear():
    """test linear function returns input and derivative returns
    ones in same shape as input
    """
    stim = np.random.rand(5, 7)
    assert np.all(activation("linear", stim) == stim)
    assert activation("linear", stim).shape == (5, 7)
    assert np.all(dactivation("linear", stim) == 1)
    assert dactivation("linear", stim).shape == (5, 7)
