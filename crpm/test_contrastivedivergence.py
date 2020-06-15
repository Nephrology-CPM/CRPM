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
    from crpm.ffn_bodyplan import stack_new_layer
    from crpm.gradientdecent import gradientdecent
    #init numpy seed
    np.random.seed(2860486313)

    #setup model
    model, data = setup_nestedcs()

    #remove discriminating layer
    prototype = model[0:-1]

    #explicitly remove labels from data
    #labels = data[2, :]
    data = data[0:2, :]

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(prototype, data, maxepoch=0)

    #calculate initial mean squared error
    pred, icost = computecost(autoencoder, data, data, "mse")

    #train model
    _, autoencoder = contrastivedivergence(prototype, data,
                                           ncd=10,
                                           nadj=10,
                                           maxepoch=1000,
                                           momentum=0.1,
                                           batchsize=10,
                                           finetune=6)

    #calculate final mean squared error
    pred, cost = computecost(autoencoder, data, data, "mse")

    #assert learning is taking place
    assert icost > cost

def test_encode_periodiccases_deep():
    """test periodiccases can be encoded
    """
    import numpy as np
    from crpm.setup_periodiccases import setup_periodiccases_deep
    from crpm.contrastivedivergence import contrastivedivergence
    from crpm.dynamics import computecost

    #init numpy seed
    np.random.seed(2860486313)

    #setup model
    model, data = setup_periodiccases_deep()
    nx = data.shape[0]
    nsample = data.shape[1]

    #remove discriminating layer
    prototype = model[0:-1]

    #partition training and validation data
    valid = data[1:nx,0:nsample//3]
    #validtargets = data[0,0:nsample//3]
    train = data[1:nx,nsample//3:nsample]
    #targets =data[0,nsample//3:nsample]

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(prototype, train, maxepoch=0)

    #calculate initial reconstruction error
    pred, icost = computecost(autoencoder, valid, valid, "mse")

    #train prototype
    _, autoencoder = contrastivedivergence(prototype, train, validata=valid,
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

def r_test_encode_spectra2():
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
    np.random.seed(2860486313)

    #setup model
    model, data = setup_spectra2()

    #remove discriminating layer
    prototype = model[0:-1]

    #partition data (labels on first row)
    nobv = data.shape[1]
    cutoff = 2*nobv//3
    #target = data[0, :cutoff]
    train = data[1:, :cutoff]
    #vtarget = data[0, cutoff:]
    valid = data[1:, cutoff:]

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(prototype, train, maxepoch=0)

    #calculate initial reconstruction error
    pred, icost = computecost(autoencoder, valid, valid, "mse")
    print("init recon error = " + str(icost))

    #train prototype
    #_, autoencoder = contrastivedivergence(prototype, train, validata=valid,
    #                                       ncd=1,
    #                                       batchsize=50,
    #                                       nadj=10,
    #                                       maxepoch=100,
    #                                       momentum=0.0)
    _, autoencoder = contrastivedivergence(prototype, train, validata=valid,
                                           ncd=1,
                                           batchsize=10,
                                           nadj=10,
                                           maxepoch=1000,
                                           momentum=0.9,
                                           finetune=7)

    #calculate final reconstruction error
    pred, cost = computecost(autoencoder, valid, valid, "mse")
    print("pretrained recon error = " + str(cost))

    #assert learning is taking place
    assert icost > cost

def r_test_pretrain_periodiccases_deep():
    """test pretained periodiccases model encodes better than non pretrained model
    """
    import numpy as np
    from crpm.setup_periodiccases import setup_periodiccases_deep
    from crpm.dynamics import computecost

    from crpm.gradientdecent import gradientdecent
    from crpm.ffn_bodyplan import reinit_ffn
    from crpm.contrastivedivergence import contrastivedivergence
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier
    from crpm.ffn_bodyplan import stack_new_layer
    from crpm.ffn_bodyplan import copy_ffn
    #from crpm.analyzebinaryclassifier import plotroc

    #init numpy seed
    np.random.seed(2860486313)

    #setup model
    model, data = setup_periodiccases_deep()
    nx = data.shape[0]
    nsample = data.shape[1]

    #partition training and validation data
    valid = data[1:nx,0:nsample//3]
    validtargets = data[0,0:nsample//3]
    train = data[1:nx,nsample//3:nsample]
    targets =data[0,nsample//3:nsample]

    #remove discriminating layer
    prototype = model[0:-1]

    #re-init prototype
    prototype = reinit_ffn(prototype)

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(prototype, train, maxepoch=0)

    #calculate initial reconstruction error
    pred, icost_encoder = computecost(autoencoder, valid, valid, "mse")

    #conventional training autoencoder
    pred, gcost_encoder, _ = gradientdecent(autoencoder, train, train, "mse",
                                    valid, valid,
                                    maxepoch=1E6,
                                    earlystop=True,
                                    healforces=False,
                                    finetune=9)

    #assert auto encoder can be conventionally trained
    assert gcost_encoder<icost_encoder

    #re-init prototype
    prototype = reinit_ffn(prototype)

    #CD train autoencoder
    _, autoencoder = contrastivedivergence(prototype, train, validata=valid,
                                           ncd=10,
                                           batchsize=20,
                                           nadj=10,
                                           maxepoch=500,
                                           momentum=0.05,
                                           finetune=6)

    #calculate reconstruction error
    pred, cost_encoder = computecost(autoencoder, valid, valid, "mse")
    print(cost_encoder)

    #assert reconstruction error is less than initial recon error
    assert cost_encoder < icost_encoder

    #fine-tune autoencoder
    pred, fcost_encoder, _ = gradientdecent(autoencoder, train, train, "mse",
                                    valid, valid,
                                    maxepoch=1E6,
                                    earlystop=True,
                                    healforces=False,
                                    finetune=9)

    #assert final reconstruction error is not greater than previous recon error
    assert fcost_encoder <= cost_encoder

    #assert final reconstruction error is not greater than with conventional training
    assert fcost_encoder <= gcost_encoder
