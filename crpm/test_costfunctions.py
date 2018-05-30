""" Test cost functions
"""

import numpy as np
from crpm.costfunctions import loss

def test_mse():
    """test mean square error
    """
    pred = np.ones((3, 5))
    targets = np.array([1, 1, 2, 3, 6])
    dif = pred - targets
    cost, dloss = loss("mse", pred, targets)
    assert cost == 3
    assert np.all(dloss == dif)
