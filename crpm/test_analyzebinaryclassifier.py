""" test binary classifier
"""
def test_solved_nestedcs():
    """test solved nestedcs will produce optimal threashold
    """
    import numpy as np
    from crpm.setup_nestedcs import setup_nestedcs
    from crpm.gradientdecent import gradientdecent
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier

    #init numpy seed
    np.random.seed(1500450271)

    #setup model
    model, data = setup_nestedcs()

    #train model
    pred, _, _ = gradientdecent(model, data[0:2,], data[-1,], "bce")

    #analyze binary classifier
    _, report = analyzebinaryclassifier(pred, data[-1,])

    print(report)
    assert report["AreaUnderCurve"] >= .95
    assert report["Accuracy"] >= .85
    assert report["Accuracy"] < 1.0
    assert report["F1Score"] >= .85

def will_roc_will_plot():
    """ test if roc output will plot properly"""
    import os
    import matplotlib.pyplot as plt
    from crpm.setup_nestedcs import setup_nestedcs
    from crpm.gradientdecent import gradientdecent
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier

    #setup model
    model, data = setup_nestedcs()

    #train model
    pred, _, _ = gradientdecent(model, data[0:2,], data[-1,], "bce")

    #analyze binary classifier
    roc, _ = analyzebinaryclassifier(pred, data[-1,])

    plt.scatter(*zip(*roc))

    #remove any previously saved files if they exist
    if os.path.exists("nestedcs_roc.png"):
        os.remove("nestedcs_roc.png")

    #save file
    plt.savefig("nestedcs_roc.png")
    #plt.show()

    #assert file was created then remove
    assert os.path.exists("nestedcs_roc.png")
    os.remove("nestedcs_roc.png")
