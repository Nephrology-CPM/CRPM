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

    print(model[1]["weight"])

    assert np.allclose(model[1]["weight"], 1.0, rtol=.0015)

def test_solve_nestedcs():
    """test nested cs can be solved
    """

    from crpm.setup_nestedcs import setup_nestedcs
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.gradientdecent import gradientdecent

    #setup model
    model, data = setup_nestedcs()

    #calculate initial mean squared error
    pred, __ = fwdprop(data[0:2,], model)
    icost, __ = loss("mse", pred, data[-1,])
    #print(icost)

    #train model
    pred, cost = gradientdecent(model, data[0:2,], data[-1,], "mse")

    print(model)
    assert icost > cost
    assert cost < .046

def test_solve_nestedcs_bce():
    """test nested cs can be solved
    """

    from crpm.setup_nestedcs import setup_nestedcs
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.gradientdecent import gradientdecent

    #setup model
    model, data = setup_nestedcs()

    #calculate initial mean squared error
    pred, __ = fwdprop(data[0:2,], model)
    icost, __ = loss("bce", pred, data[-1,])

    #train model
    pred, cost = gradientdecent(model, data[0:2,], data[-1,], "bce")

    print(model)
    print(icost)
    print(cost)
    assert icost > cost
    assert cost < .27
