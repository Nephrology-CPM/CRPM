""" test contrastive divergence training algorithm
"""

def test_encode_nestedcs():
    """test nested cs data can be encoded
    """

    import numpy as np
    from crpm.setup_nestedcs import setup_nestedcs
    from crpm.contrastivedivergence import contrastivedivergence
    from crpm.dynamics import computecost
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier
    from crpm.analyzebinaryclassifier import plotroc
    from crpm.ffn_bodyplan import stack_new_layer
    from crpm.gradientdecent import gradientdecent
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt


    #init numpy seed
    np.random.seed(40017)

    #setup model
    model, data = setup_nestedcs()

    #remove discriminating layer
    prototype = model[0:-1]

    #explicitly remove labels from data
    labels = data[2, :]
    data = data[0:2, :]


    #zscore data
    data = np.divide(data-np.mean(data, axis=1, keepdims=True),
                     np.std(data, axis=1, keepdims=True))

    #analyze untrained binary classifier
    pred, cost = computecost(model, data, labels, "bce")
    roc, report = analyzebinaryclassifier(pred, labels)
    if report["AreaUnderCurve"]<.5:
        #flip labels
        pred, cost = computecost(model, data, 1-labels, "bce")
        roc, report = analyzebinaryclassifier(pred, labels)
    print(report)
    plotroc(roc)

    cases = np.where(labels==1,True, False)
    p2 = np.reshape(np.where(pred>.4818,True, False).T,(200,))
    print(p2.shape)
    print(cases.shape)
    plt.scatter(data[0,cases],data[1,cases])
    plt.scatter(data[0,p2],data[1,p2])

    left off here



    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(prototype, data, maxepoch=0)

    #calculate initial mean squared error
    pred, icost = computecost(autoencoder, data, data, "mse")

    #train model
    #_, autoencoder = contrastivedivergence(model, data, ncd=10, maxepoch=100, momentum=0.1)
    _, autoencoder = contrastivedivergence(prototype, data,
                                           ncd=1,
                                           nadj=10,
                                           maxepoch=100,
                                           momentum=0.0,
                                           batchsize=1,
                                           finetune=6)


    #calculate final mean squared error
    pred, cost = computecost(autoencoder, data, data, "mse")

    #diagnostic
    print(icost)
    print(cost)

    #assert learning is taking place
    #assert icost > cost

    #create discriminator
    discriminator = stack_new_layer(prototype, n=1, activation="logistic")

    #train model (no pre-training)
    pred, cost, _ = gradientdecent(discriminator, data, labels, "bce", finetune=6)

    #analyze fine-tuned binary classifier
    pred, cost = computecost(discriminator, data, labels, "bce")
    roc, report = analyzebinaryclassifier(pred, labels)
    if report["AreaUnderCurve"]<.5:
        #flip labels
        pred, cost = computecost(discriminator, data, 1-labels, "bce")
        roc, report = analyzebinaryclassifier(pred, labels)
    print(report)
    plotroc(roc)

    assert False

def test_encode_periodiccases():
    """test periodiccases can be encoded
    """
    import numpy as np
    from crpm.setup_periodiccases import setup_periodiccases
    from crpm.contrastivedivergence import contrastivedivergence
    from crpm.dynamics import computecost

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

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(model, train, maxepoch=0)

    #calculate initial reconstruction error
    pred, icost = computecost(autoencoder, valid, valid, "mse")

    #train model
    _, autoencoder = contrastivedivergence(model, train, validata=valid,
                                           ncd=10,
                                           nadj=20,
                                           maxepoch=2000,
                                           momentum=0.9,
                                           batchsize=20,
                                           finetune=7)

    #calculate final reconstruction error
    pred, cost = computecost(autoencoder, valid, valid, "mse")

    #print(autoencoder)
    print(icost)
    print(cost)

    #assert learning is taking place
    assert icost > cost
    assert False

def test_encode_periodiccases_deep():
    """test periodiccases can be encoded
    """
    import numpy as np
    from crpm.setup_periodiccases import setup_periodiccases_deep
    from crpm.contrastivedivergence import contrastivedivergence
    from crpm.dynamics import computecost

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
    pred, icost = computecost(autoencoder, valid, valid, "mse")

    #train model
    _, autoencoder = contrastivedivergence(model, train, validata=valid,
                                           ncd=5,
                                           maxepoch=500,
                                           momentum=.05)

    #calculate final reconstruction error
    pred, cost = computecost(autoencoder, valid, valid, "mse")

    #diagnostic
    print(icost)
    print(cost)

    #assert learning is taking place
    assert icost > cost

def test_pretrain_periodiccases_deep():
    """test pretained periodiccases model encodes better than non pretrained model
    """
    import numpy as np
    from crpm.setup_periodiccases import setup_periodiccases_deep
    from crpm.dynamics import computecost

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

    #analyze untrained binary classifier
    pred, icost = computecost(model, valid, validtargets, "bce")
    roc, ireport = analyzebinaryclassifier(pred, validtargets)
    if ireport["AreaUnderCurve"]<.5:
        #flip labels
        pred, icost = computecost(model, valid, 1-validtargets, "bce")
        roc, ireport = analyzebinaryclassifier(pred, validtargets)
    print(ireport)
    #plotroc(roc)

    #train model (no pre-training)
    pred, cost, _ = gradientdecent(model, train, targets, "bce",
                                   valid, validtargets, earlystop=True)

    #analyze trained binary classifier
    pred, cost = computecost(model, valid, validtargets, "bce")
    roc, report = analyzebinaryclassifier(pred, validtargets)
    if report["AreaUnderCurve"]<.5:
        #flip labels
        pred, cost = computecost(model, valid, 1-validtargets, "bce")
        roc, report = analyzebinaryclassifier(pred, validtargets)
    print(report)
    #plotroc(roc)

    #re-init model
    model = reinit_ffn(model)

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(model, train, maxepoch=0)

    #calculate initial reconstruction error
    pred, icost_encoder = computecost(autoencoder, valid, valid, "mse")

    #train autoencoder
    _, autoencoder = contrastivedivergence(model, train, validata=valid,
                                           ncd=5,
                                           maxepoch=500,
                                           momentum=.05)

    #calculate final reconstruction error
    pred, cost_encoder = computecost(autoencoder, valid, valid, "mse")


    #analyze pre-trained binary classifier
    pred, mcost = computecost(model, valid, validtargets, "bce")
    roc, mreport = analyzebinaryclassifier(pred, validtargets)
    if report["AreaUnderCurve"]<.5:
        #flip labels
        pred, mcost = computecost(model, valid, 1-validtargets, "bce")
        roc, mreport = analyzebinaryclassifier(pred, validtargets)
    print(mreport)
    #plotroc(roc)

    #fine-tune model
    pred, fcost, _ = gradientdecent(model, train, targets, "bce",
                                    valid, validtargets, earlystop=True,
                                    healforces=False)

    #analyze fine-tuned binary classifier
    pred, fcost = computecost(model, valid, validtargets, "bce")
    roc, freport = analyzebinaryclassifier(pred, validtargets)
    if report["AreaUnderCurve"]<.5:
        #flip labels
        pred, fcost = computecost(model, valid, 1-validtargets, "bce")
        roc, freport = analyzebinaryclassifier(pred, validtargets)
    print(freport)
    #plotroc(roc)

    print("===================")
    print("Untrained cost = " + str(icost))
    print("AUC = " + str(ireport["AreaUnderCurve"]))
    print("- - - - - - - - - -")
    print("Standard-training cost = " + str(cost))
    print("AUC = " + str(report["AreaUnderCurve"]))
    print("- - - - - - - - - -")
    print("intermediate cost" + str(mcost))
    print("AUC = " + str(mreport["AreaUnderCurve"]))
    print("- - - - - - - - - -")
    print("final cost" + str(fcost))
    print("AUC = " + str(freport["AreaUnderCurve"]))
    print("===================")
    print("untrained reconstruction error" + str(icost_encoder))
    print("CD-training reconstruction error" + str(cost_encoder))
    print("===================")

    #assert fine tuning after pretraining is better than standard-training
    assert cost > fcost
    #assert contrastivedivergence improves prediction
    assert report["AreaUnderCurve"] < mreport["AreaUnderCurve"]
    #assert fine tuning prediction is at least is as good as CD
    assert  mreport["AreaUnderCurve"] <= freport["AreaUnderCurve"]

def test_encode_spectra2():
    """test spectra2 can be encoded
    """

    import numpy as np
    from crpm.setup_spectra2 import setup_spectra2
    from crpm.lossfunctions import loss
    from crpm.gradientdecent import gradientdecent
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier
    from crpm.contrastivedivergence import contrastivedivergence
    from crpm.ffn import FFN
    from crpm.dynamics import computecost

    from crpm.fwdprop import fwdprop

    #init numpy seed
    np.random.seed(40017)

    #setup model
    prototype, data = setup_spectra2()

    #get prototype depth
    nlayer = len(prototype)

    #partition data (labels on first row)
    nobv = data.shape[1]
    cutoff = 2*nobv//3
    target = data[0, :cutoff]
    train = data[1:, :cutoff]
    vtarget = data[0, cutoff:]
    valid = data[1:, cutoff:]

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(prototype, train, maxepoch=0)

    #calculate initial reconstruction error
    pred, icost = computecost(autoencoder, valid, valid, "mse")
    print("init recon error = " + str(icost))

    ##train prototype
    #_, autoencoder = contrastivedivergence(prototype, train,
    #                                       ncd=2,
    #                                       batchsize=50,
    #                                       nadj=10,
    #                                       maxepoch=100,
    #                                       momentum=0.1)
    #train prototype
    _, autoencoder = contrastivedivergence(prototype, train, validata=valid,
                                           ncd=5,
                                           batchsize=20,
                                           nadj=10,
                                           maxepoch=2000,
                                           momentum=0.9)

    #calculate final reconstruction error
    pred, cost = computecost(autoencoder, valid, valid, "mse")
    print("pretrained recon error = " + str(cost))

    #assert learning is taking place
    assert icost > cost
