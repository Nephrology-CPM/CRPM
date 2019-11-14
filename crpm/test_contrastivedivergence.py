""" test contrastive divergence training algorithm
"""

def test_encode_nestedcs():
    """test nested cs data can be encoded
    """

    import numpy as np
    from crpm.setup_nestedcs import setup_nestedcs
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.contrastivedivergence import contrastivedivergence

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_nestedcs()

    #explicitly remove labels from data
    data = data[0:2,]

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(model, data, maxepoch=0)

    #calculate initial mean squared error
    pred, _ = fwdprop(data, autoencoder)
    icost, _ = loss("mse", pred, data)

    #train model
    _, autoencoder = contrastivedivergence(model, data, ncd=10, maxepoch=100, momentum=0.1)

    #calculate final mean squared error
    pred, _ = fwdprop(data,autoencoder)
    cost, _ = loss("mse", pred, data)

    print(autoencoder)
    print(icost)
    print(cost)

    #assert learning is taking place
    assert icost > cost

def test_encode_periodiccases():
    """test periodiccases can be encoded
    """
    import numpy as np
    from crpm.setup_periodiccases import setup_periodiccases
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.contrastivedivergence import contrastivedivergence

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_periodiccases()
    nx = data.shape[0]
    nsample = data.shape[1]

    #partition training and validation data
    valid = data[1:nx,0:nsample//3]
    validtargets = data[0,0:nsample//3]
    train = data[1:nx,nsample//3:nsample]
    targets =data[0,nsample//3:nsample]

    #targets = data[0, 0:nsample]
    #data = data[1:nx, 0:nsample]

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(model, train, maxepoch=0)

    #calculate initial reconstruction error
    pred, _ = fwdprop(train, autoencoder)
    icost, _ = loss("mse", pred, train)

    #train model
    _, autoencoder = contrastivedivergence(model, train, ncd=1, maxepoch=200, momentum=.5)

    #calculate final reconstruction error
    pred, _ = fwdprop(train, autoencoder)
    cost, _ = loss("mse", pred, train)

    #print(autoencoder)
    print(icost)
    print(cost)

    #assert learning is taking place
    assert icost > cost

def test_encode_periodiccases_deep():
    """test periodiccases can be encoded
    """
    import numpy as np
    from crpm.setup_periodiccases import setup_periodiccases_deep
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.contrastivedivergence import contrastivedivergence

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_periodiccases_deep()
    nx = data.shape[0]
    nsample = data.shape[1]

    #partition training and validation data
    valid = data[1:nx,0:nsample//3]
    validtargets = data[0,0:nsample//3]
    train = data[1:nx,nsample//3:nsample]
    targets =data[0,nsample//3:nsample]

    #targets = data[0, 0:nsample]
    #data = data[1:nx, 0:nsample]

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(model, train, maxepoch=0)

    #calculate initial reconstruction error
    pred, _ = fwdprop(train, autoencoder)
    icost, _ = loss("mse", pred, train)

    #train model
    _, autoencoder = contrastivedivergence(model, train, ncd=1, maxepoch=200, momentum=0.0)

    #calculate final reconstruction error
    pred, _ = fwdprop(train, autoencoder)
    cost, _ = loss("mse", pred, train)

    #print(autoencoder)
    print(icost)
    print(cost)

    #assert learning is taking place
    assert icost > cost

def test_pretrain_periodiccases():
    """test pretained periodiccases model encodes better than non pretrained model
    """
    import numpy as np
    from crpm.setup_periodiccases import setup_periodiccases
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.gradientdecent import gradientdecent
    from crpm.ffn_bodyplan import reinit_ffn
    from crpm.contrastivedivergence import contrastivedivergence
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier
    from crpm.analyzebinaryclassifier import plotroc

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_periodiccases()
    nx = data.shape[0]
    nsample = data.shape[1]

    #partition training and validation data
    valid = data[1:nx,0:nsample//3]
    validtargets = data[0,0:nsample//3]
    train = data[1:nx,nsample//3:nsample]
    targets =data[0,nsample//3:nsample]

    #calculate initial binary cross entropy error
    pred, _ = fwdprop(valid, model)
    icost, _ = loss("bce", pred, validtargets)
    roc, report = analyzebinaryclassifier(pred, validtargets)
    if report["AreaUnderCurve"]<.5:
        pred = 1-pred
        icost, _ = loss("bce", pred, validtargets)
        roc, report = analyzebinaryclassifier(pred, validtargets)
    print(report)
    #plotroc(roc)

    #train model (no pre-training)
    pred, cost, _ = gradientdecent(model, train, targets, "bce", valid, validtargets, earlystop=True)
    print("no pre-training cost")
    print(cost)

    #calculate out-sample analysis
    pred, _ = fwdprop(valid, model)
    roc, report = analyzebinaryclassifier(pred, validtargets)
    if report["AreaUnderCurve"]<.5:
        pred = 1-pred
        roc, report = analyzebinaryclassifier(pred, validtargets)
    print(report)
    #plotroc(roc)

    #re-init model
    model = reinit_ffn(model)

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(model, train, maxepoch=0)

    #calculate initial reconstruction error
    pred, _ = fwdprop(valid, autoencoder)
    icost_encoder, _ = loss("mse", pred, valid)

    #pre-train model
    _, autoencoder = contrastivedivergence(model, train, ncd=1, maxepoch=100, momentum=.5)

    #calculate final reconstruction error
    pred, _ = fwdprop(valid, autoencoder)
    cost_encoder, _ = loss("mse", pred, valid)

    #calculate intermediate binary cross entropy error
    pred, _ = fwdprop(valid, model)
    mcost, _ = loss("bce", pred, validtargets)
    roc, mreport = analyzebinaryclassifier(pred, validtargets)
    if mreport["AreaUnderCurve"]<.5:
        pred = 1-pred
        mcost, _ = loss("bce", pred, validtargets)
        roc, mreport = analyzebinaryclassifier(pred, validtargets)
    print(mreport)
    #plotroc(roc)

    #fine-tune model
    pred, fcost, _ = gradientdecent(model, train, targets, "bce", valid, validtargets, earlystop=True)

    #calculate final out-sample analysis
    roc, freport = analyzebinaryclassifier(pred, validtargets)
    if report["AreaUnderCurve"]<.5:
        pred = 1-pred
        fcost, _ = loss("bce", pred, validtargets)
        roc, freport = analyzebinaryclassifier(pred, validtargets)
    print(freport)
    #plotroc(roc)

    print("initial cost")
    print(icost)
    print("no pre-training cost")
    print(cost)
    print("intermediate cost")
    print(mcost)
    print("final cost")
    print(fcost)

    print("initial reconstruction error")
    print(icost_encoder)
    print("pre-training reconstruction error")
    print(cost_encoder)

    #assert learning is taking place
    assert icost > fcost

    #assert contrastivedivergence or fine tuning helps
    CDhelps = report["AreaUnderCurve"] > mreport["AreaUnderCurve"]
    finetunehelps = report["AreaUnderCurve"] > freport["AreaUnderCurve"]
    assert CDhelps or finetunehelps

def test_pretrain_periodiccases_deep():
    """test pretained periodiccases model encodes better than non pretrained model
    """
    import numpy as np
    from crpm.setup_periodiccases import setup_periodiccases_deep
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.gradientdecent import gradientdecent
    from crpm.ffn_bodyplan import reinit_ffn
    from crpm.contrastivedivergence import contrastivedivergence
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier
    from crpm.analyzebinaryclassifier import plotroc

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_periodiccases_deep()
    nx = data.shape[0]
    nsample = data.shape[1]

    #partition training and validation data
    valid = data[1:nx,0:nsample//3]
    validtargets = data[0,0:nsample//3]
    train = data[1:nx,nsample//3:nsample]
    targets =data[0,nsample//3:nsample]

    #calculate initial binary cross entropy error
    pred, _ = fwdprop(valid, model)
    icost, _ = loss("bce", pred, validtargets)
    roc, report = analyzebinaryclassifier(pred, validtargets)
    if report["AreaUnderCurve"]<.5:
        pred = 1-pred
        icost, _ = loss("bce", pred, validtargets)
        roc, report = analyzebinaryclassifier(pred, validtargets)
    print(report)
    #plotroc(roc)

    #train model (no pre-training)
    pred, cost, _ = gradientdecent(model, train, targets, "bce", valid, validtargets, earlystop=True)
    print("no pre-training cost")
    print(cost)

    #calculate out-sample analysis
    pred, _ = fwdprop(valid, model)
    roc, report = analyzebinaryclassifier(pred, validtargets)
    if report["AreaUnderCurve"]<.5:
        pred = 1-pred
        roc, report = analyzebinaryclassifier(pred, validtargets)
    print(report)
    #plotroc(roc)

    #re-init model
    model = reinit_ffn(model)

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(model, train, maxepoch=0)

    #calculate initial reconstruction error
    pred, _ = fwdprop(valid, autoencoder)
    icost_encoder, _ = loss("mse", pred, valid)

    #pre-train model
    _, autoencoder = contrastivedivergence(model, train, ncd=1, maxepoch=200, momentum=0.0)

    #calculate final reconstruction error
    pred, _ = fwdprop(valid, autoencoder)
    cost_encoder, _ = loss("mse", pred, valid)

    #calculate intermediate binary cross entropy error
    pred, _ = fwdprop(valid, model)
    mcost, _ = loss("bce", pred, validtargets)
    roc, report = analyzebinaryclassifier(pred, validtargets)
    if report["AreaUnderCurve"]<.5:
        pred = 1-pred
        mcost, _ = loss("bce", pred, validtargets)
        roc, report = analyzebinaryclassifier(pred, validtargets)
    print(report)
    #plotroc(roc)

    #fine-tune model
    pred, fcost, _ = gradientdecent(model, train, targets, "bce", valid, validtargets, earlystop=True)

    #calculate final out-sample analysis
    roc, report = analyzebinaryclassifier(pred, validtargets)
    if report["AreaUnderCurve"]<.5:
        pred = 1-pred
        fcost, _ = loss("bce", pred, validtargets)
        roc, report = analyzebinaryclassifier(pred, validtargets)
    print(report)
    #plotroc(roc)

    print("initial cost")
    print(icost)
    print("no pre-training cost")
    print(cost)
    print("intermediate cost")
    print(mcost)
    print("final cost")
    print(fcost)

    print("initial reconstruction error")
    print(icost_encoder)
    print("pre-training reconstruction error")
    print(cost_encoder)

    #assert fine tuning after pretraining is better than without pre-training
    assert cost > fcost

def test_stability_periodiccases_deep():
    """test stability of periodiccases deep model by contrastivedivergence
    """
    import numpy as np
    from crpm.setup_periodiccases import setup_periodiccases_deep
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.contrastivedivergence import contrastivedivergence

    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_periodiccases_deep()
    nx = data.shape[0]
    nsample = data.shape[1]

    #partition training and validation data
    valid = data[1:nx,0:nsample//3]
    validtargets = data[0,0:nsample//3]
    train = data[1:nx,nsample//3:nsample]
    targets =data[0,nsample//3:nsample]

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(model, train, maxepoch=0)

    #calculate initial reconstruction error
    pred, _ = fwdprop(train, autoencoder)
    icost, _ = loss("mse", pred, train)

    #train model
    _, autoencoder = contrastivedivergence(model, train, ncd=1, maxepoch=200, momentum=0.0)

    #calculate final reconstruction error
    pred, _ = fwdprop(train, autoencoder)
    cost, _ = loss("mse", pred, train)

    #print(autoencoder)
    print(icost)
    print(cost)

    #assert learning is taking place
    assert icost > cost
