""" test gradient decent training algorithm
"""

def test_solve_numberadder():
    """test number adder can be solved begining with weights = 1.1
    """
    import numpy as np
    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset
    from crpm.gradientdecent import gradientdecent

    #create shallow bodyplan with 5 inputs and 1 output for numebr adder data
    bodyplan = read_bodyplan("crpm/data/shallowfornumberadder.csv")

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

    import numpy as np
    from crpm.setup_nestedcs import setup_nestedcs
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_nestedcs()

    #calculate initial mean squared error
    pred, __ = fwdprop(data[0:2,], model)
    icost, __ = loss("mse", pred, data[-1,])
    #print(icost)

    #train model
    pred, cost = gradientdecent(model, data[0:2,], data[-1,], "mse")

    #print(model)
    #print(icost)
    #print(cost)
    assert icost > cost
    assert cost < .046

def test_solve_nestedcs_bce():
    """test nested cs can be solved
    """
    import numpy as np
    from crpm.setup_nestedcs import setup_nestedcs
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_nestedcs()

    #calculate initial binary cross entropy error
    pred, __ = fwdprop(data[0:2,], model)
    icost, __ = loss("bce", pred, data[-1,])

    #train model
    pred, cost = gradientdecent(model, data[0:2,], data[-1,], "bce")

    #print(model)
    #print(icost)
    #print(cost)
    assert icost > cost
    assert cost < .27

def test_regval_reduces_uncorrel():
    """ test that regularization term will reduce the weight assoctiated with
    uncorrelated features compared with no regularization. Example function is
    y = x1 - x2 + x3^2, where features x1, x2, and x3 are indepently sampled from
    normal distribution with zero mean and unit variance."""

    import numpy as np
    from crpm.setup_multicorrel import setup_multicorrel
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(40017)

    #setup model with no regularization
    model, data = setup_multicorrel()

    #get data dimensions
    nvar = data.shape[0]
    nobv = data.shape[1]

    # partition training and testing data
    train = data[0:3,0:nobv//2]
    target = data[-1,0:nobv//2]
    vtrain = data[0:3,nobv//2:nobv]
    vtarget = data[-1,nobv//2:nobv]

    #train model with mean squared error
    __, icost = gradientdecent(model, train, target, "mse", validata = vtrain, valitargets = vtarget)

    #save weights
    iweight = model[1]["weight"]

    #re-init model
    model, data = setup_multicorrel()
    #manually set regularization term
    model[1]["regval"] = 75

    #train regularized model
    __, cost = gradientdecent(model, train, target, "mse", validata = vtrain, valitargets = vtarget)

    #save weights
    weight = model[1]["weight"]


    #print(iweight.shape)
    #print(weight)
    #norm = np.linalg.norm(iweight)
    #if norm > 0: iweight = iweight / norm
    #norm = np.linalg.norm(weight)
    #if norm > 0: weight = weight / norm
    #print(iweight)
    #print(weight)
    #print(icost)
    #print(cost)
    assert abs(iweight[0,2]) > abs(weight[0,2])
    #assert icost > cost
