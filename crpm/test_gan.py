""" test generative adversarial network training algorithm
"""

def r_test_spectra2():
    """test spectra2 can be encoded and generated
    """

    import numpy as np
    from crpm.setup_spectra2 import setup_spectra2
    from crpm.dynamics import computecost
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier
    #from crpm.lossfunctions import loss
    #from crpm.analyzebinaryclassifier import plotroc
    from crpm.gradientdecent import gradientdecent
    from crpm.contrastivedivergence import contrastivedivergence
    #from crpm.ffn import FFN
    from crpm.ffn_bodyplan import stack_new_layer
    from crpm.ffn_bodyplan import copy_ffn
    from crpm.fwdprop import fwdprop
    from crpm.backprop import backprop
    #from crpm.dynamics import computeforces
    #from crpm.dynamics import maxforce

    from crpm.gan import gan
    #import matplotlib
    #matplotlib.use('TkAgg')
    #import matplotlib.pyplot as plt

    #init numpy seed
    np.random.seed(3628273133)

    #setup model
    prototype, data = setup_spectra2()

    #get prototype depth
    nlayer = len(prototype)

    #get data dimensions
    nfeat = data.shape[0]
    nobv = data.shape[1]

    #zscore data
    tdata = np.divide(data-np.mean(data, axis=1, keepdims=True),np.std(data, axis=1, keepdims=True))

    #transform features into boltzmann like probs
    #tdata = np.exp(-data)
    #partfunc = np.sum(tdata, axis=1, keepdims = True)
    #tdata = np.divide(tdata,partfunc) #normalize
    #tdata = np.divide(tdata, np.max(tdata, axis=1, keepdims=True))#scale features by maxintensity

    #plt.plot(data[:,0])
    #plt.show()
    #plt.plot(tdata[:,0])
    #plt.show()
    #data = tdata

    #partition data (labels on first row)
    ntrain = 2*nobv//3
    target = data[0, :ntrain]
    train = data[1:, :ntrain]
    vtarget = data[0, ntrain:]
    valid = data[1:, ntrain:]

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(prototype, train, maxepoch=0)

    #calculate initial reconstruction error
    pred, ireconerr = computecost(autoencoder, valid, valid, "mse")
    print("init recon error = " + str(ireconerr))

    ##train prototype
    #_, autoencoder = contrastivedivergence(prototype, train,
    #                                       ncd=2,
    #                                       batchsize=50,
    #                                       nadj=10,
    #                                       maxepoch=100,
    #                                       momentum=0.1)
    #train prototype
    _, autoencoder = contrastivedivergence(prototype, train, validata=valid,
                                           ncd=1,
                                           batchsize=50,
                                           nadj=10,
                                           maxepoch=100,
                                           momentum=0.0)

    #calculate final reconstruction error
    pred, reconerr = computecost(autoencoder, valid, valid, "mse")
    print("pretrained recon error = " + str(reconerr))

    #assert learning is taking place by reduced recon error.
    assert ireconerr > reconerr


    # ----- Discriminator -----
    #create discriminator
    discriminator = copy_ffn(autoencoder[0:len(prototype)])
    discriminator = stack_new_layer(discriminator, n=1, activation="logistic")
    #analyze trained binary classifier
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
                                   valid, vtarget, earlystop=True,
                                   finetune=6)

    #analyze trained binary classifier
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

    #assert discriminator has potential to iden two calsses
    assert report["AreaUnderCurve"] > ireport["AreaUnderCurve"]
    #assert report["AreaUnderCurve"] > .6

    # ----- generator -----

    #create generator from decoder
    generator = copy_ffn(autoencoder[len(prototype):len(autoencoder)])

    #adjust regularization
    for layer in generator:
        layer["regval"] = 0#.00001

    #correct label idecies
    idx = 0
    for layer in generator:
        generator[idx]["layer"] = idx
        idx += 1

    #generate fake samples
    nfake = 600
    ncode = generator[0]["n"]
    fake, _ = fwdprop(np.random.rand(ncode, nfake),generator)

    #calculate initial reconstruction error
    pred, fkreconerr = computecost(autoencoder, fake, fake, "mse")
    print("init fake recon error = " + str(fkreconerr))

    #assert fake data recon error is better than untrained recon error
    assert fkreconerr < ireconerr

    #-- Start GAN training---

    ganerr = gan(generator, discriminator, train, valid,
                 maxepoch=20000, batchsize=50, finetune=6.3)

    #assert generator fools discriminator at least 40% of the time.
    assert ganerr[-1,1] <-np.log(.40)

    #def moving_average(a, n=3) :
    #    ret = np.cumsum(a, dtype=float)
    #    ret[n:] = ret[n:] - ret[:-n]
    #    return ret[n - 1:] / n


    #fig = plt.figure()
    #plt.plot(ganerr[:, 0], ganerr[:, 1])
    #plt.plot(moving_average(ganerr[:, 0], n=20), moving_average(ganerr[:, 1], n=20))
    #plt.plot(ganerr[0, 0], ganerr[0, 1], marker="D", color="green", markersize=10)
    #plt.plot(ganerr[-1, 0], ganerr[-1, 1], marker="8", color="red", markersize=10)
    #plt.xlabel("discriminator error")
    #plt.ylabel("generator error")
    #plt.show()

    #print("final report")
    #print(report)
    #plotroc(roc)

    #assert False

def test_afnetwork():
    """test AF network patients can be encoded and generated
    """
    #import matplotlib
    #matplotlib.use('TkAgg')
    #import matplotlib.pyplot as plt
    #import matplotlib.patches as mpatches

    import numpy as np
    from crpm.setup_afmodel import setup_afmodel

    from crpm.dynamics import computecost
    from crpm.analyzebinaryclassifier import analyzebinaryclassifier
    #from crpm.lossfunctions import loss
    from crpm.analyzebinaryclassifier import plotroc
    from crpm.gradientdecent import gradientdecent
    from crpm.contrastivedivergence import contrastivedivergence
    #from crpm.ffn import FFN
    from crpm.ffn_bodyplan import stack_new_layer
    from crpm.ffn_bodyplan import copy_ffn
    from crpm.fwdprop import fwdprop
    #from crpm.backprop import backprop
    #from crpm.dynamics import computeforces
    #from crpm.dynamics import maxforce

    from crpm.gan import gan

    #init numpy seed
    np.random.seed(3628273133)

    #setup model
    prototype, train, target, valid, vtarget = setup_afmodel()

    #trim data
    #maxobv = 150
    #train = train[:,:maxobv]
    #valid = valid[:,:maxobv]
    #target = target[:maxobv]
    #vtarget = vtarget[:maxobv]

    #get prototype depth
    nlayer = len(prototype)

    #get data dimensions
    nfeat = train.shape[0]
    nobv = train.shape[1]

    #return untrained autoencoder
    _, autoencoder = contrastivedivergence(prototype, train, maxepoch=0)

    # ----- Discriminator -----

    #create discriminator
    discriminator = copy_ffn(autoencoder[0:len(prototype)])
    discriminator = stack_new_layer(discriminator, n=1, activation="logistic")

    print("analyze untrained discriminator to iden subtype")
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
    #assert icost > cost

    #assert discriminator has potential to iden two classes
    #assert report["AreaUnderCurve"] > ireport["AreaUnderCurve"]
    #assert report["AreaUnderCurve"] > .55

    # ----- GENERATOR -----

    #create generator from decoder
    generator = copy_ffn(autoencoder[len(prototype)-1:len(autoencoder)])

    #correct label idecies
    idx = 0
    for layer in generator:
        generator[idx]["layer"] = idx
        idx += 1

    #assert False
    #-- Main GAN training---
    #ganerr = gan(generator, discriminator, train,
    #                   maxepoch=100000, batchsize=1, finetune=6)
    ganerr = gan(generator, discriminator, train, valid,
                       maxepoch=100000, batchsize=1, finetune=6)

    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    ganerr[:,2] = np.log(ganerr[:,2]) #plot density error on logscale
    discerrbar = moving_average(ganerr[:, 0], n=20)
    generrbar = moving_average(ganerr[:, 1], n=20)
    autoerrbar = moving_average(ganerr[:, 2], n=20)

    #assert generator fools discriminator at least 40% of the time.
    print(ganerr[-1,1])
    assert ganerr[-1,1] <-np.log(.40)

    #fig = plt.figure()
    #plt.plot(ganerr[:, 0], ganerr[:, 1])
    #plt.plot(discerrbar, generrbar)
    #plt.plot(discerrbar[0], generrbar[0], marker="D", color="green", markersize=10)
    #plt.plot(discerrbar[-1], generrbar[-1], marker="8", color="red", markersize=10)
    #plt.xlabel("discriminator error")
    #plt.ylabel("generator error")
    #plt.show()

    #fig = plt.figure()
    #plt.plot(ganerr[:, 0], ganerr[:, 2])
    #plt.plot(discerrbar, autoerrbar)
    #plt.plot(discerrbar[0], autoerrbar[0], marker="D", color="green", markersize=10)
    #plt.plot(discerrbar[-1], autoerrbar[-1], marker="8", color="red", markersize=10)
    #plt.xlabel("discriminator error")
    #plt.ylabel("encoder error")
    #plt.show()

    #generate fake data for every training sample
    nsample = train.shape[1]
    fake, _ = fwdprop(np.random.rand(generator[0]["n"], nsample), generator)
    #merge training and fake data
    gandata = np.hstack((train, fake))
    ganlabels = np.hstack((np.repeat(1, nsample),np.repeat(0, nsample)))

    print("analyze trained discriminator on fake vs training set")
    pred, cost = computecost(discriminator, gandata, ganlabels, "bce")
    roc, report = analyzebinaryclassifier(pred, ganlabels)
    if report["AreaUnderCurve"]<.5:
        #flip labels
        pred, cost = computecost(discriminator, gandata, ganlabels, "bce")
        roc, report = analyzebinaryclassifier(pred, 1-ganlabels)
    print(report)
    #plotroc(roc)

    #gen fake data for every validation sample
    nsample = valid.shape[1]
    fake, _ = fwdprop(np.random.rand(generator[0]["n"], nsample), generator)
    #merge validation and fake data
    gandata = np.hstack((valid, fake))
    ganlabels = np.hstack((np.repeat(1, nsample),np.repeat(0, nsample)))

    print("analyze trained discriminator on fake vs vaidation set")
    pred, costv = computecost(discriminator, gandata, ganlabels, "bce")
    roc, reportv = analyzebinaryclassifier(pred, ganlabels)
    if reportv["AreaUnderCurve"]<.5:
        #flip labels
        pred, costv = computecost(discriminator, gandata, 1-ganlabels, "bce")
        roc, reportv = analyzebinaryclassifier(pred, 1-ganlabels)
    print(reportv)
    #plotroc(roc)

    #assert discriminator has poor potential to iden fake data
    #assert reportv["AreaUnderCurve"] <.55

    #get fake data the discriminator thinks is real
    pred, _ = fwdprop(fake, discriminator)
    spoof = fake[:, pred[0, :] > report["OptimalThreshold"]]

    #plot metabolite distributions
    labels = []
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    #add_label(plt.violinplot(train.T), "Training")
    ##add_label(plt.violinplot(valid.T), "Validation")
    #add_label(plt.violinplot(fake.T), "Simulated")
    ##add_label(plt.violinplot(spoof.T), "Spoofed")

    #plt.legend(*zip(*labels))


    #plt.show()

    #assert False
