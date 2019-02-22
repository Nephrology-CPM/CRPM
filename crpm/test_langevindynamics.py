""" test Langevin dynamics training algorithm
"""

maxepoch = int(2E3) #5E4
maxbuffer = int(1E2) #1E2

def test_solve_numberadder():
    """test number adder can be solved begining with init weights set
    """
    import numpy as np
    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.dataset import load_dataset
    from crpm.ffn_bodyplan import init_ffn
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.langevindynamics import langevindynamics

    initweight = 1.5

    #load data
    __, data = load_dataset("crpm/data/numberadder.csv")
    __, testdata = load_dataset("crpm/data/numberadder_test.csv")

    #create shallow bodyplan with 5 inputs and 1 output for numebr adder data
    bodyplan = read_bodyplan("crpm/data/numberadder_bodyplan.csv")

    #create numberadder model
    model = init_ffn(bodyplan)

    #manually set layer weights to 1.1 and biases to 0
    model[1]["weight"] = initweight*np.ones(model[1]["weight"].shape)

    #calculate initial mean squared error
    pred, __ = fwdprop(data[0:5,], model)
    icost, __ = loss("mse", pred, data[-1,])
    print("icost = "+str(icost))
    print(model[1]["weight"])

    #train numberadder model  with mean squared error
    __, cost = langevindynamics(model, data[0:5,], data[-1,], "mse", testdata[0:5,], testdata[-1,], maxepoch=int(3E5), maxbuffer=int(1E3))
    print("cost ="+str(cost))
    print(model[1]["weight"])

    assert icost>cost
    assert np.allclose(model[1]["weight"], 1.0, rtol=.005)

def test_solve_nestedcs():
    """test nested cs can be solved
    """

    import numpy as np
    from crpm.setup_nestedcs import setup_nestedcs
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.langevindynamics import langevindynamics

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_nestedcs()

    #calculate initial mean squared error
    pred, __ = fwdprop(data[0:2,], model)
    icost, __ = loss("mse", pred, data[-1,])
    #print(icost)

    #train model
    pred, cost = langevindynamics(model, data[0:2,], data[-1,], "mse", maxepoch=maxepoch, maxbuffer=maxbuffer)

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
    from crpm.langevindynamics import langevindynamics

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_nestedcs()

    #calculate initial binary cross entropy error
    pred, __ = fwdprop(data[0:2,], model)
    icost, __ = loss("bce", pred, data[-1,])

    #train model
    pred, cost = langevindynamics(model, data[0:2,], data[-1,], "bce", maxepoch=maxepoch, maxbuffer=maxbuffer)

    #print(model)
    #print(icost)
    #print(cost)
    assert icost > cost
    assert cost < .29
