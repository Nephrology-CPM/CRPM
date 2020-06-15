""" generative adversarial network training algorithm
"""

def gan(generator, discriminator, data, valid=None, maxepoch=500, nout=100,
        batchsize=10, finetune=6, sigma0=0, label0=1, withdecay= True):
    """ Trains generative adversarial network by semi gradientdecent.
        Args:
            data: training data with features in rows and observations in columns
            valid: optional validation data in same format as training data
            generator: ffn model with number of nodes in output layer equal to
                the number of features in the training data.
            discriminator: ffn model with sigle node logistic in the output layer
                and number of nodes in the input layer equal to the number of
                features in the training data.
            maxepoch: maximum number of training steps.
            nout: number of steps reported.
            batchsize: size of minibatch for SGD training.
            finetune: tuning parameter that scales inversely with learning step.
        Returns:
        An numpy array with nout rows and 3 columns representing the
         discriminator error, generator error, and reconstruction error.
    """

    import numpy as np
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.backprop import backprop
    from crpm.dynamics import computeforces
    from crpm.dynamics import maxforce
    from crpm.ffn_bodyplan import copy_ffn

    #partition training data if no validation data is provided
    if valid is None:
        #get number of training samples
        nobv = data.shape[1]
        #throw error and exit if 25% of observations is less than 5
        if nobv//4 < 5:
            print("Data must contain at least 20 observations.")
            return None
        #select 25% observations from data
        sel = np.random.choice(nobv, size=nobv//4, replace=False)
        #partition data
        valid = data[:, sel]
        data = np.delete(data, sel, axis=1)

    #recalculate data dimensions
    nfeat = data.shape[0]
    nobv = data.shape[1]
    nvalid = valid.shape[1]

    # ----- check input -----

    def isnotpositiveint(var):
        """ will return true if var is not a positive integer"""
        if not isinstance(var, int):
            return True
        if var <= 0:
            return True
        return False

    #check discriminator has logistic output
    if discriminator[-1]["activation"] != "logistic":
        print("Warning: discriminator should have logistic output.")
        return None
    #check discriminator has single node output
    if discriminator[-1]["n"] != 1:
        print("Warning: discriminator should output a single number.")
        return None
    #check generator outputs a value for all features in the training data
    if generator[-1]["n"] != nfeat:
        print("Warning: number of nodes in generator ouptut layer should be " +
              "equal to number of rows in data.")
        return None
    #check generator has linear or logistic input
    if (generator[0]["activation"] != "linear" and
        generator[0]["activation"] != "logistic"):
        print("Warning: generator must have linear or logistic input.")
        return None
    #check discriminator penultimate layer has same size and activation as generator input layer
    if (discriminator[-2]["activation"] != generator[0]["activation"] or
        discriminator[-2]["n"] != generator[0]["n"]):
        print("Warning: discriminator penultimate layer must match generator input layer.")
        return None
    #check for positive number of training steps
    if isnotpositiveint(maxepoch):
        #throw error msg and return nothing
        print("Warning maxepoch is not a positive integer!")
        return None
    #check for positive number of training steps
    if isnotpositiveint(batchsize):
        #throw error msg and return nothing
        print("Warning batchsize is not a positive integer!")
        return None

    #-- Start GAN training---

    #set output frequency
    delay = 0
    if(maxepoch>nout):
        nadj = maxepoch//nout
        delay = maxepoch%nadj
    else:
        nadj = maxepoch

    #init ganerr record for True Positive logodds, False Negative logodds, recon mse, and epoch
    ganerr = np.empty((nout+1, 4))

    #img noise decay
    sigma_fac = sigma0
    sigma_decay = sigma_fac/nout #linear decay

    #label smoothing decay
    lsmooth = label0
    lsmooth_rate = (1-lsmooth)/nout

    #correct minibatch size if larger than number of observations in data
    minibatch = min(batchsize, nobv)

    #learning rate regulator
    alpha_norm = 10**(-finetune)

    #get number of generator encoding nodes
    ncode = generator[0]["n"]

    #loop over epochs
    for epoch in range(maxepoch+1):

        #select mini batch from data
        sel = np.random.choice(nobv, size=minibatch, replace=False)

        #sample mini batch of noise
        if(generator[0]["activation"]=="linear"):
            #sample gaussian distribution
            noise = np.random.randn(ncode, minibatch)
        if(generator[0]["activation"]=="logistic"):
            #sample uniform distribution
            noise = np.random.rand(ncode, minibatch)

        #sample image noise
        sigma = data[:, sel].std(axis=1, keepdims=True)*sigma_fac
        imgnoise = np.random.randn(data.shape[0], minibatch) * sigma
        fkimgnoise = np.random.randn(data.shape[0], minibatch) * sigma

        # - - Train discriminator to detect real data:
        #     increase TPR (decr T1err)

        #compute forces on discriminator
        pred, discstate = fwdprop(data[:, sel]+imgnoise, discriminator)
        #derr, dloss = loss("bce", pred, np.repeat(1, minibatch))
        derr, dloss = loss("bce", pred,
                           np.repeat(lsmooth, minibatch), #label smoothing
                           logit=discstate[-1]["stimulus"])
        forces, _ = backprop(discriminator, discstate, dloss)

        #normalize learning rate alpha based on current forces
        alpha = alpha_norm * maxforce(discriminator, forces)

        #update discriminator weights and biases
        for layer in forces:
            index = layer["layer"]
            discriminator[index]["weight"] = (discriminator[index]["weight"] +
                                              alpha * layer["fweight"])
            discriminator[index]["bias"] = (discriminator[index]["bias"] +
                                            alpha * layer["fbias"])

        # - - Train generator to reproduce discriminator latent representation:
        #     autoencoding to increase FNR? (incr T2err?)
        #     should improve mode collapse

        #fwd prop encoder(discriminator upto penultimate layer) state
        latent, encstate = fwdprop(data[:, sel], discriminator[:-1])
        #fwd prop decoder(generator) state
        recon, genstate = fwdprop(latent, generator)

        #compute autoencoder reconstruction error
        autoerr, dloss = loss("mse", recon, data[:, sel])

        #compute forces on decoder(generator)
        forces, _ = backprop(generator, genstate, dloss)

        #normalize learning rate alpha based on current forces
        alpha = alpha_norm * maxforce(generator, forces) / 4

        #update decoder weights and biases
        for layer in forces:
            index = layer["layer"]
            generator[index]["weight"] = (generator[index]["weight"] +
                                          alpha * layer["fweight"])
            generator[index]["bias"] = (generator[index]["bias"] +
                                        alpha * layer["fbias"])

        # - - Train discriminator to detect fake data:
        #     increase TNR (decr T2err)

        # generate fake data
        fake, genstate = fwdprop(noise, generator)

        #compute forces on discriminator
        pred, discstate = fwdprop(fake+fkimgnoise, discriminator)
        #derr, dloss = loss("bce", pred, np.repeat(0, minibatch))
        derr, dloss = loss("bce", pred,
                           np.repeat(1-lsmooth, minibatch),
                           logit=discstate[-1]["stimulus"])
        forces, _ = backprop(discriminator, discstate, dloss)

        #normalize learning rate alpha based on current forces
        alpha = alpha_norm * maxforce(discriminator, forces)

        #update discriminator weights and biases
        for layer in forces:
            index = layer["layer"]
            discriminator[index]["weight"] = (discriminator[index]["weight"] +
                                              alpha * layer["fweight"])
            discriminator[index]["bias"] = (discriminator[index]["bias"] +
                                            alpha * layer["fbias"])

        # - - Train generator to fool discriminator:
        #     increase FPR (incr T1err)

        # generate fake data
        fake, genstate = fwdprop(noise, generator)

        # compute discriminator state due to fake data
        pred, discstate = fwdprop(fake+fkimgnoise, discriminator)

        # calculate derivative of missclassification error
        #gerr, dloss = loss("bce", pred, np.repeat(1, minibatch))
        gerr, dloss = loss("bce", pred,
                           np.repeat(1, minibatch),
                           logit=discstate[-1]["stimulus"])

        # back prop gradient on generator coming from disccr missclassification
        _, dact = backprop(discriminator, discstate, dloss)

        # get forces on generator
        forces, _ = backprop(generator, genstate, dact)

        # normalize learning rate alpha based on current forces
        alpha = alpha_norm * maxforce(generator, forces) / 4

        # update body weights and biases
        for layer in forces:
            index = layer["layer"]
            generator[index]["weight"] = (generator[index]["weight"] +
                                          alpha * layer["fweight"])
            generator[index]["bias"] = (generator[index]["bias"] +
                                        alpha * layer["fbias"])

        # - - Periodic Validation:
        # adjust label smoothing and image noise factors

        #book keeping
        idx = epoch-delay
        if idx%nadj == 0 and idx >= 0:

            #sample noise for every sample in validation
            if(generator[0]["activation"]=="linear"):
                #sample gaussian distribution
                vnoise = np.random.randn(ncode, nvalid)
            if(generator[0]["activation"]=="logistic"):
                #sample uniform distribution
                vnoise = np.random.rand(ncode, nvalid)

            # generate fake data
            fake, genstate = fwdprop(vnoise, generator)

            #calc recon error on validation set
            latent, encstate = fwdprop(valid, discriminator[:-1])
            recon, genstate = fwdprop(latent, generator)
            vautoerr, _ = loss("mse", recon, valid)

            #calc disc error on validation set (True Positive Rate)
            pred, discstate = fwdprop(valid, discriminator)
            vderr, _ = loss("bce", pred, np.repeat(1, nvalid))

            #calc gen error on fake set (False Positive Rate)
            pred, discstate = fwdprop(fake, discriminator)
            vgerr, _ = loss("bce", pred, np.repeat(1, nvalid))

            #save learning error
            ganerr[idx//nadj, :] = [vderr, vgerr, vautoerr, epoch]
            print(str(epoch) + ": " +
                  str((np.exp(-2*vderr)+np.exp(-2*vgerr))/2) + ": " +
                  str(np.exp(-vderr)/np.exp(-vgerr)) + ": " +
                  str(np.log(vautoerr)))

            #adjust factors if decay is on
            if withdecay:
                #adjust noise factor
                sigma_fac -= sigma_decay
                #adjust label smoothing
                lsmooth += lsmooth_rate

    return ganerr
