""" Test forward propagation methods for FFNs.
"""

def test_fwdprop_number_adder():
    """test that unit weights will make a number adder.
    """
    import numpy as np
    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset
    from crpm.fwdprop import fwdprop

    #create shallow bodyplan with 5 inputs and 1 output for numebr adder data
    bodyplan = read_bodyplan("crpm/data/shallowfornumberadder.csv")

    #create model
    model = init_ffn(bodyplan)

    #manually set layer 1 weights to 1 and biases to 0
    model[1]["weight"] = np.ones(model[1]["weight"].shape)

    #run forward propagation with example data in number_adder.csv
    __, data = load_dataset("crpm/data/number_adder.csv")
    indepvars = data[0:5,]
    depvars = data[-1,]
    prediction, __ = fwdprop(indepvars, model)

    assert np.all(depvars == prediction)
