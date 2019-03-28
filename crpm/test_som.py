""" test self oranizing map algorithm
"""

def test_solve_nestedcs():
    """test nested cs can be solved
    """

    import numpy as np
    from crpm.setup_nestedcs import setup_nestedcs
    from crpm.som import som
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_nestedcs()

    #conduct mapping
    pred, _ = som(model, data[0:2,])

    #analyze binary classifier
    _, report = analyzebinaryclassifier(pred, data[-1,])

    assert report["AreaUnderCurve"] >= .8 #.92
    assert report["Accuracy"] >= .8 #.84
    assert report["F1Score"] >= .76 #.85
