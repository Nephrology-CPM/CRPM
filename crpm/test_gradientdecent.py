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

def test_solve_toruscases_bce():
    """test toruscases can be solved
    """
    import numpy as np
    from crpm.setup_toruscases import setup_toruscases
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.gradientdecent import gradientdecent
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_toruscases()
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

    #analyze binary classifier
    pred, _ = fwdprop(valid, model)
    roc, ireport = analyzebinaryclassifier(pred, validtargets)
    if ireport["AreaUnderCurve"]<.5:
        pred = 1-pred
        icost, _ = loss("bce", pred, validtargets)
        roc, ireport = analyzebinaryclassifier(pred, validtargets)
    print(ireport)
    #plotroc(roc)

    #train model
    pred, cost, _ = gradientdecent(model, train, targets, "bce", valid, validtargets, earlystop=True)

    #analyze binary classifier
    pred, _ = fwdprop(valid, model)
    roc, report = analyzebinaryclassifier(pred, validtargets)
    if report["AreaUnderCurve"]<.5:
        pred = 1-pred
        cost, _ = loss("bce", pred, validtargets)
        roc, report = analyzebinaryclassifier(pred, validtargets)
    print(report)
    #plotroc(roc)


    #print(model)
    print(icost)
    print(cost)
    assert icost > cost
    assert cost < .4
    assert report["MatthewsCorrCoef"] > .1
    #don't expect problem can be solved with linear model
    #assert report["AreaUnderCurve"] > ireport["AreaUnderCurve"]

def rtest_solve_toruscases_deep_bce():
    """test toruscases_deep can be solved
    """
    import numpy as np
    from crpm.setup_toruscases import setup_toruscases_deep
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.gradientdecent import gradientdecent
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_toruscases_deep()
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

    #analyze binary classifier
    pred, _ = fwdprop(valid, model)
    roc, ireport = analyzebinaryclassifier(pred, validtargets)
    if ireport["AreaUnderCurve"]<.5:
        pred = 1-pred
        icost, _ = loss("bce", pred, validtargets)
        roc, ireport = analyzebinaryclassifier(pred, validtargets)
    print(ireport)
    #plotroc(roc)

    #train model
    pred, cost, _ = gradientdecent(model, train, targets, "bce", valid, validtargets, earlystop=True)

    #analyze binary classifier
    pred, _ = fwdprop(valid, model)
    roc, report = analyzebinaryclassifier(pred, validtargets)
    if report["AreaUnderCurve"]<.5:
        pred = 1-pred
        cost, _ = loss("bce", pred, validtargets)
        roc, report = analyzebinaryclassifier(pred, validtargets)
    print(report)
    #plotroc(roc)

    #print(model)
    print(icost)
    print(cost)

    #assert learning has taken place
    assert icost > cost

    #assert cost is less than 1.7
    assert cost < 1.7

    assert report["AreaUnderCurve"] > ireport["AreaUnderCurve"]
    assert report["AreaUnderCurve"] > .8


def test_classify_spectra2():
    """test spectra2 can find two groups
    """

    import numpy as np
    from crpm.setup_spectra2 import setup_spectra2
    from crpm.dynamics import computecost
    from crpm.gradientdecent import gradientdecent
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier

    #init numpy seed
    np.random.seed(40017)

    #setup model
    discriminator, data = setup_spectra2()

    #partition data (labels on first row)
    nobv = data.shape[1]
    cutoff = 2*nobv//3
    target = data[0, :cutoff]
    train = data[1:, :cutoff]
    vtarget = data[0, cutoff:]
    valid = data[1:, cutoff:]

    #analyze untrained discriminator
    pred, icost = computecost(discriminator, valid, vtarget, "bce")
    roc, ireport = analyzebinaryclassifier(pred, vtarget)
    if ireport["AreaUnderCurve"]<.5:
        #flip labels
        pred, icost = computecost(discriminator, valid, 1-vtarget, "bce")
        roc, ireport = analyzebinaryclassifier(pred, 1-vtarget)
    print(ireport)
    #plotroc(roc)

    #train discriminator
    pred, cost, _ = gradientdecent(discriminator, train, target, "bce",
                                   valid, vtarget,
                                   earlystop=True,
                                   finetune=7)

    #analyze discriminator
    print("analyze trained discriminator to iden subtype")
    pred, cost = computecost(discriminator, valid, vtarget, "bce")
    roc, report = analyzebinaryclassifier(pred, vtarget)
    if report["AreaUnderCurve"]<.5:
        #flip labels
        pred, cost = computecost(discriminator, valid, 1-vtarget, "bce")
        roc, report = analyzebinaryclassifier(pred, 1-vtarget)
    print(report)
    #plotroc(roc)

    #assert discriminator can be trained by binary cross entropy error
    assert icost > cost
    assert report["AreaUnderCurve"] > ireport["AreaUnderCurve"]
    assert report["AreaUnderCurve"] > .8
