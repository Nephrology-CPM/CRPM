""" Test back propagation methods for FFNs.
"""

def test_backprop_number_adder():
    """test that solved number adder will have zero forces with proper shape.
    """
    import numpy as np
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.backprop import backprop

    #manually create shallow bodyplan for number_adder.csv data
    bodyplan = [{"layer":0, "n":5, "activation":"linear"}]
    bodyplan.append({"layer":1, "n":1, "activation":"linear"})

    #create number_adder model
    addermodel = init_ffn(bodyplan)

    #manually set layer 1 weights to 1 and biases to 0
    addermodel[1]["weight"] = np.ones(addermodel[1]["weight"].shape)

    #compute forces using number_adder.csv data with mean squared error
    __, data = load_dataset("crpm/data/number_adder.csv")
    pred, state = fwdprop(data[0:5,], addermodel)
    __, dloss = loss("mse", pred, data[-1,])
    forces = backprop(addermodel, state, dloss)

    assert forces[-1]["dweight"].shape == (1, 5)
    assert np.all(forces[-1]["dweight"] == 0)
    assert forces[-1]["dbias"].shape == (1, 1)
    assert np.all(forces[-1]["dbias"] == 0)

def test_numadd_forcedir():
    """test that number adder with initial wieghts >1 will have negative forces.
    """
    import numpy as np
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.backprop import backprop

    #manually create shallow bodyplan for number_adder.csv data
    bodyplan = [{"layer":0, "n":5, "activation":"linear"}]
    bodyplan.append({"layer":1, "n":1, "activation":"linear"})

    #create number_adder model
    addermodel = init_ffn(bodyplan)

    #manually set layer 1 weights to 1.1 and biases to 0
    addermodel[1]["weight"] = 1.1 * np.ones(addermodel[1]["weight"].shape)

    #compute forces using number_adder.csv data with mean squared error
    __, data = load_dataset("crpm/data/number_adder.csv")
    pred, state = fwdprop(data[0:5,], addermodel)
    __, dloss = loss("mse", pred, data[-1,])
    forces = backprop(addermodel, state, dloss)

    assert np.all(forces[-1]["dweight"] < 0)
    assert np.all(forces[-1]["dbias"] < 0)
