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
    bodyplan = read_bodyplan("crpm/data/numberadder_bodyplan.csv")

    #create numberadder model
    model = init_ffn(bodyplan)

    #manually set layer weights to 1.1 and biases to 0
    model[1]["weight"] = 1.1*np.ones(model[1]["weight"].shape)

    #train numberadder model  with mean squared error
    _, data = load_dataset("crpm/data/numberadder.csv")
    _, _, _ = gradientdecent(model, data[0:5,], data[-1,], "mse")

    print(model[1]["weight"])

    assert np.allclose(model[1]["weight"], 1.0, rtol=.005)

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
    pred, _ = fwdprop(data[0:2,], model)
    icost, _ = loss("mse", pred, data[-1,])
    #print(icost)

    #train model
    pred, cost, _ = gradientdecent(model, data[0:2,], data[-1,], "mse")

    #print(model)
    #print(icost)
    #print(cost)
    assert icost > cost
    assert cost < .08

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
    pred, _ = fwdprop(data[0:2,], model)
    icost, _ = loss("bce", pred, data[-1,])

    #train model
    pred, cost, _ = gradientdecent(model, data[0:2,], data[-1,], "bce")

    #print(model)
    #print(icost)
    #print(cost)
    assert icost > cost
    assert cost < .29

def test_solve_periodiccases_bce():
    """test periodiccases can be solved
    """
    import numpy as np
    from crpm.setup_periodiccases import setup_periodiccases
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_periodiccases()
    nx = data.shape[0]
    nsample = data.shape[1]

    #partition training and validation data
    valid = data[1:data.shape[0],0:nsample//3]
    validtargets = data[0,0:nsample//3]
    train = data[1:data.shape[0],nsample//3:nsample]
    targets =data[0,nsample//3:nsample]

    #calculate initial binary cross entropy error
    pred, _ = fwdprop(train, model)
    icost, _ = loss("bce", pred, targets)

    #train model
    pred, cost, _ = gradientdecent(model, train, targets, "bce", valid, validtargets, earlystop=True)

    print(model)
    print(icost)
    print(cost)
    assert icost > cost
    assert cost < .71

def test_solve_periodiccases_deep_bce():
    """test periodiccases_deep can be solved
    """
    import numpy as np
    from crpm.setup_periodiccases import setup_periodiccases_deep
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.gradientdecent import gradientdecent

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_periodiccases_deep()
    nx = data.shape[0]
    nsample = data.shape[1]

    #partition training and validation data
    valid = data[1:data.shape[0],0:nsample//3]
    validtargets = data[0,0:nsample//3]
    train = data[1:data.shape[0],nsample//3:nsample]
    targets =data[0,nsample//3:nsample]

    #calculate initial binary cross entropy error
    pred, _ = fwdprop(train, model)
    icost, _ = loss("bce", pred, targets)

    #train model
    pred, cost, _ = gradientdecent(model, train, targets, "bce", valid, validtargets, earlystop=True)

    print(model)
    print(icost)
    print(cost)

    #assert learning has taken place
    assert icost > cost

    #assert cost is less than 1.7
    assert cost < 1.7
