""" test gradient decent training algorithm
"""

def test_solve_numberadder():
    """test number adder can be solved begining with weights = 1.1
    """
    import numpy as np
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset
    from crpm.gradientdecent import gradientdecent

    #manually create shallow bodyplan for number_adder.csv data
    bodyplan = [{"layer":0, "n":5, "activation":"linear"}]
    bodyplan.append({"layer":1, "n":1, "activation":"linear"})

    #create number_adder model
    model = init_ffn(bodyplan)

    #manually set layer weights to 1.1 and biases to 0
    model[1]["weight"] = 1.1*np.ones(model[1]["weight"].shape)

    #train number_adder model  with mean squared error
    __, data = load_dataset("crpm/data/number_adder.csv")
    __, __ = gradientdecent(model, data[0:5,], data[-1,], "mse")

    assert np.allclose(model[1]["weight"], 1.0, rtol=.001)

def test_solve_nestedcs():
    """test nested cs can be solved
    """
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset
    from crpm.gradientdecent import gradientdecent
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier

    #manually create a bodyplan for nestedCs.csv data
    bodyplan = [
        {"layer":0, "n":2, "activation":"linear"},
        {"layer":2, "n":1, "activation":"logistic"}
        ]

    #create model
    model = init_ffn(bodyplan)

    #download nestedCs data
    __, data = load_dataset("crpm/data/nestedCs.csv")

    #train model
    pred, __ = gradientdecent(model, data[0:2,], data[-1,], "mse")

    #analyze binary classifier
    __, report = analyzebinaryclassifier(pred, data[-1,])

    #print(report)
    #import matplotlib.pyplot as plt
    #plt.scatter(*zip(*ROC))
    #plt.show()

    assert report["Accuracy"] > .80
