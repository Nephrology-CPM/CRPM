""" test generative adversarial network training algorithm
"""

def gan(generator, discriminator, data, maxepoch=500, batchsize=10, finetune=6):
    """ Trains generative adversarial network by semi gradientdecent.
        Args:
            data: training data with features in rows and observations in columns
            generator: ffn model with number of nodes in output layer equal to
                the number of features in the training data.
            discriminator: ffn model with sigle node logistic in the output layer
                and number of nodes in the input layer equal to the number of
                features in the training data.
            maxepoch: optional maximum number of training steps.
            batchsize: optional size of minibatch for SGD training.
            finetune: tuning parameter that scales inversely with learning step.
        Returns:
        cost: discriminator final binary cross entropy error
    """

    import numpy as np
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.backprop import backprop
    from crpm.dynamics import computeforces
    from crpm.dynamics import maxforce
    from crpm.ffn_bodyplan import copy_ffn

    #get data dimensions
    nfeat = data.shape[0]
    nobv = data.shape[1]

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

    #save only 5k-10k points
    nadj = 1
    delay = 0
    maxpts = 5000
    if(maxepoch>maxpts):
        nadj = maxepoch//maxpts
        delay = maxepoch%maxpts
    else:
        maxpts = maxepoch

    #init ganerr record for discriminator bce, generator bce, encoder mse, and epoch
    ganerr = np.empty((maxpts, 4))

    #init best disc and gen models
    #best_discriminator = copy_ffn(discriminator)
    #best_generator = copy_ffn(generator)
    #besterr = None

    #correct minibatch size if larger than number of observations in data
    minibatch = min(batchsize, nobv)

    #learning rate regulator
    alpha_norm = 10**(-finetune)

    #get number of generator encoding nodes
    ncode = generator[0]["n"]

    ##select initial 1/2 batch from data
    #sel = np.random.choice(nobv, size=minibatch, replace=False)
    ##sample initial 1/2 batch of noise
    #noise = np.random.rand(ncode, minibatch)

    #loop over epochs
    for epoch in range(maxepoch):

        #select mini batch from data
        sel = np.random.choice(nobv, size=minibatch, replace=False)
        #sample mini batch of noise
        noise = np.random.rand(ncode, minibatch)

        # - - Train discriminator to detect real data:
        #     increase TPR (decr T1err)

        #compute forces on discriminator
        pred, discstate = fwdprop(data[:,sel], discriminator)
        derr, dloss = loss("bce", pred, np.repeat(1, minibatch))
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
        latent, encstate = fwdprop(data[:,sel], discriminator[:-1])
        #fwd prop decoder(generator) state
        recon, genstate = fwdprop(latent, generator)

        #compute autoencoder reconstruction error
        autoerr, dloss = loss("mse", recon, data[:,sel])

        #compute forces on decoder(generator)
        forces, _ = backprop(generator, genstate, dloss)

        #normalize learning rate alpha based on current forces
        alpha = alpha_norm * maxforce(generator, forces)

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
        pred, discstate = fwdprop(fake, discriminator)
        derr, dloss = loss("bce", pred, np.repeat(0, minibatch))
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

        # compute discriminator state due to fake data
        pred, discstate = fwdprop(fake, discriminator)

        # calculate derivative of missclassification error
        gerr, dloss = loss("bce", pred, np.repeat(1, minibatch))

        # back prop gradient on generator coming from disccr missclassification
        _, dact = backprop(discriminator, discstate, dloss)

        # get forces on generator
        forces, _ = backprop(generator, genstate, dact)

        # normalize learning rate alpha based on current forces
        alpha = alpha_norm * maxforce(generator, forces)

        # update body wieghts and biases
        for layer in forces:
            index = layer["layer"]
            generator[index]["weight"] = (generator[index]["weight"] +
                                          alpha * layer["fweight"])
            generator[index]["bias"] = (generator[index]["bias"] +
                                        alpha * layer["fbias"])

        #save best autoencoding discriminator-generator pair
        #if besterr is None:
        #    besterr = autoerr
        #if autoerr < besterr:
        #    best_discriminator = copy_ffn(discriminator)
        #    best_generator = copy_ffn(generator)

        #book keeping
        idx = epoch-delay
        if idx%nadj == 0 and idx >= 0:
            ganerr[idx//nadj, :] = [derr, gerr, autoerr, epoch]

    #Overwrite discriminator and generator
    #discriminator = copy_ffn(best_discriminator)
    #generator = copy_ffn(best_generator)

    return ganerr
