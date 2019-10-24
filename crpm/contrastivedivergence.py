""" NN training by contrastive divergence
"""

def contrastivedivergence(model, data, N=1, maxepoch=100, nadj=10, momentum=.5, batchsize=10, validata=None):
    """unfold and train fnn model by contrastive divergence

        Args:
            model: deep FFN model
            data: features in rows, observations in columns.
            N: number of contrastive divergence steps
            maxepoch: hard limit of learning iterations default is 100
            nadj: period of learning rate adjustment in units of epochs
            momentum: fraction of previous change in weight carried over to
                      next weight update step
        Returns: exit condition and trained unfolded model.
            Exit conditions are 0) learning converged, 1) learning not
            converged, and -1) learning cannot be performed.
            Training will modify model.
    """

    import numpy as np
    from crpm.activationfunctions import activation
    from crpm.ffn_bodyplan import get_bodyplan
    from crpm.ffn_bodyplan import copy_bodyplan
    from crpm.ffn_bodyplan import push_bodyplanlayer
    from crpm.ffn_bodyplan import init_ffn

    #init exit condition to default
    exitcond = 0

    #get model bodyplan
    bodyplan = get_bodyplan(model)
    #get number of model layers
    nlayer = len(model)
    #copy bodyplan
    unfolded_bodyplan = copy_bodyplan(bodyplan)
    #push layers in reversed order to create a symmetric bodyplan
    for layer in reversed(bodyplan[:-1]):
        push_bodyplanlayer(unfolded_bodyplan, layer)
    #create unfolded model from symmetric bodyplan
    smodel = init_ffn(unfolded_bodyplan)
    #print(smodel)

    #return symmetric model if maxepoch = 0
    if maxepoch<1:
        return exitcond, smodel

    #define minibatches
    #get number of observations in data
    nobv = data.shape[1]
    #calculate number of minibatches needed
    batchsize = int(batchsize)
    nbatch = nobv//batchsize
    #get randomized observation index
    data = data.T
    np.random.shuffle(data)
    data = data.T

    #konstant: scales learning rate by max force relative to weight
    alpha_norm = 5E-6

    #initialize previous layer activity with input data for layer 0
    prevlayeractivity = data

    #do the same for the validation data
    validprevlayeractivity = validata
    if validata is None:
        #use last 20% of batches for validation
        vbatch = nbatch//5
        nbatch = nbatch - vbatch
        prevlayeractivity = data[:, 0:nbatch*batchsize]
        validprevlayeractivity = data[:, nbatch*batchsize:]

    # loop over first half of symmetric model begining with layer 1
    for layerindex in range(1, nlayer):

        #encoding index is = layerindex
        #decoding index is = 2*nlayer - layerindex +1
        decodeindex = 2*nlayer-(layerindex+1)

        #define layers
        vislayer = smodel[decodeindex]
        hidlayer = smodel[layerindex]

        #determine appropriate RBM type
        vtype = vislayer["activation"]
        htype = hidlayer["activation"]
        rbmtype = None
        #1. binary
        if vtype == "logistic" and htype == "logistic":
            rbmtype = "binary"
            #define activity for visible layer
            def vsample():
                """returns logistic visible layer activity given hiddenlayer state"""
                stimulus = np.add(hidlayer["weight"].T.dot(hstate), vislayer["bias"])
                return activation("logistic", stimulus)
            #define activity for hidden layer
            def hsample():
                """returns logistic hidden layer activity and stocastic binary state given visible layer activity"""
                stimulus = np.add(hidlayer["weight"].dot(vact), hidlayer["bias"])
                hact = activation("logistic", stimulus)
                return hact, hact > np.random.random(hact.shape)
        #2. Gaussian-Bernoulli
        if vtype == "linear" and htype == "logistic":
            rbmtype = "gaussian-bernoulli"
            #Get standard deviation for real-valued visible units
            sigma = np.std(prevlayeractivity, axis=1, keepdims=True)
            #define activity for visible layer
            def vsample():
                """returns linear plus gaussian noise visible layer activity given hidden layer state"""
                stimulus = np.add(hidlayer["weight"].T.dot(hstate)*sigma, vislayer["bias"])
                return np.random.normal(loc=stimulus, scale=sigma)
            #define activity for hidden layer
            def hsample():
                """returns logistic hidden layer activity and stocastic binary state given scaled visible layer activity"""
                stimulus = np.add(hidlayer["weight"].dot(vact/sigma), hidlayer["bias"])
                act = activation("logistic",stimulus)
                return act, act > np.random.random(act.shape)
        #3. Bernoulli-Gaussian
        if vtype == "logistic" and htype == "linear":
            rbmtype = "bernoulli-gaussian"
            #define activity for visible layer
            def vsample():
                """returns logistic visible layer activity given unit scaled hidden layer activity"""
                stimulus = np.add(hidlayer["weight"].T.dot(hstate), vislayer["bias"])
                return activation("logistic", stimulus)
            #define activity for hidden layer
            def hsample():
                """returns linear plus unit var gaussian noise hidden layer activity and stocastic state given vislayer activity"""
                stimulus = np.add(hidlayer["weight"].dot(vact),hidlayer["bias"])
                return stimulus, np.random.normal(loc=stimulus)
        #4. exit if unknown RBM type
        if rbmtype == None:
            exitcond = -1  #cannot run contrastive divergence on this model
            print("Error in contrastivedivergence.py: cannot find appropriate RBM type.")
            print("Ensure model has only logistic or linear layers.")
            print("Also ensure linear layers are not adjacent - that would be pointless btw.")
            return exitcond, smodel

        #define free energy equation
        def feng(act):
            stimulus = np.add(hidlayer["weight"].dot(act), hidlayer["bias"])
            eng = -np.sum(np.multiply(act, vislayer["bias"]))
            zeng = stimulus #for large x: log(1+exp(x)) = x
            xidx = np.where(stimulus < 12)
            zeng[idx] = np.log(1+np.exp(stimulus[idx]))
            return eng - np.sum(zeng)

        # continuous loop over learning steps (use exit conditions)
        print("training "+rbmtype+" RBM in layer "+str(layerindex))
        continuelearning = True
        epoch = 0
        err = 0
        dweight = np.zeros(hidlayer["weight"].shape)
        dhbias = np.zeros(hidlayer["bias"].shape)
        dvbias = np.zeros(vislayer["bias"].shape)
        freeeng = np.full(nadj, feng(validprevlayeractivity)
                          -feng(prevlayeractivity))
        freeeng0 = np.copy(freeeng)
        earlystop = False
        while continuelearning:
            #increment epoch counter
            epoch += 1
            #print("epoch = "+str(epoch))

            #loop over minibatches
            for batch in range(nbatch):

                #get minibatch
                minibatch = prevlayeractivity[:, batch*batchsize:(batch+1)*batchsize]

                # get visible layer activity
                vact = minibatch

                # get hidden layer activity and poshidstates
                hact, hstate = hsample()

                # get product of visible layer and hidden layer actvities
                pprod = hact.dot(vact.T)

                # get sum of visible layer activity
                pvsum = np.sum(vact, axis=1, keepdims=True)

                # get sum of hidden layer activity
                phsum = np.sum(hact, axis=1, keepdims=True)

                # loop over N Gibbs sampling iterations (at least one iteration)
                continuegibbs = True
                gibbs = 0
                while continuegibbs:
                    #increment gibbs counter
                    gibbs += 1
                    # get visible layer activity | hidden layer states
                    vact = vsample()
                    # sample hidden layer state | visible layer activity
                    hact, _ = hsample()
                    # use hidden layer activity instead of state for subsequent
                    # iterations so we overwrite hstate with the activity
                    hstate = np.copy(hact)
                    #exit condition
                    if gibbs >= N:
                        continuegibbs = False
                # get product of visible layer and hidden layer actvities
                nprod = hact.dot(vact.T)
                # get sum of visible layer activity
                nvsum = np.sum(vact, axis=1, keepdims=True)
                # get sum of hidden layer activity
                nhsum = np.sum(hact, axis=1, keepdims=True)

                # accumulate error
                err += np.sum(np.square(minibatch-vact))

                # get forces on visible layer biases
                dvbias0 = dvbias
                dvbias = (pvsum-nvsum)/batchsize

                # get forces on the hidden layer biases
                dhbias0 = dhbias
                dhbias = (phsum-nhsum)/batchsize

                #calculate forces on weights
                dweight0 = dweight
                dweight = (pprod-nprod)/batchsize
                #add regularization penalty term if specified by layer
                if hidlayer["regval"] > 0:
                    if hidlayer["lreg"] == 1:
                        dweight -= hidlayer["regval"]*np.sign(hidlayer["weight"])
                    if hidlayer["lreg"] == 2:
                        dweight -= hidlayer["regval"]*hidlayer["weight"]

                #adjust learning rate to ensure integrator doesn't break
                alpha = alpha_norm*np.max(np.divide(hidlayer["weight"], dweight))
                #print(alpha)

                #update weights with momentum term
                hidlayer["weight"] += momentum*dweight0+alpha*dweight

                # update visible layer biases with momentum term
                vislayer["bias"] += momentum*dvbias0+alpha*dvbias

                # update hidden layer biases with momentum term
                hidlayer["bias"] += momentum*dhbias0+alpha*dhbias

            # periodically check free energy for overfitting
            freeeng[epoch%nadj] = feng(validprevlayeractivity)-feng(prevlayeractivity)
            #print(np.mean(freeeng))
            if epoch%nadj == 0:
                if np.mean(freeeng) > np.mean(freeeng0)+0*np.std(freeeng0):
                    #initiate naive earlystopping
                    earlystop = True
                    print("Free engergy prev = " +str(np.mean(freeeng0)))
                    print("Free engergy curr = " +str(np.mean(freeeng)))
                freeeng0 = np.copy(freeeng)

            # - EXIT CONDITIONS -
            #exit if learning is taking too long
            if epoch > int(maxepoch):
                print("Warning contrastivedivergence.py: Training is taking a long time!"+
                      " - Try increaseing maxepoch - Training will end")
                exitcond = 1
                continuelearning = False
            #exit if naive earlystopping has been engauged
            if earlystop:
                print("Warning contrastivedivergence.py: early stopping after "
                      +str(epoch)+" epochs")
                continuelearning = False

        #symmeterize weights
        vislayer["weight"] = hidlayer["weight"].T

        #hidlayer to original model
        model[layerindex] = hidlayer

        #promote prevlayeractivity to current hidlayer activity
        vact = np.copy(prevlayeractivity)
        prevlayeractivity, _ = hsample()

        #promote validation data to current hidden layer too
        vact = np.copy(validprevlayeractivity)
        validprevlayeractivity, _ = hsample()

    # return exit condition
    return exitcond, smodel
