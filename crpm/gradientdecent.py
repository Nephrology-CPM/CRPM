""" NN training by gradient decent
"""

def model_has_bad_forces(model, data, targets, lossname):
    """ Does input data result in model forces that break integrator?"""
    import numpy as np
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.backprop import backprop

    #do one fwd-back propagation pass
    pred, state = fwdprop(data, model)
    _, dloss = loss(lossname, pred, targets)
    forces = backprop(model, state, dloss)

    #check for huge forces relative to its respective weight
    huge = 1E16
    maxf = []
    #maxw = []
    for layer in forces:
        index = layer["layer"]
        #maxf.append(np.max(abs(layer["fweight"])))
        #maxw.append(np.max(abs(model[index]["weight"])))
        maxf.append(np.max(np.abs(np.divide(layer["fweight"],model[index]["weight"]))))
    norm = np.max(maxf)

    #return True if forces are huge
    return norm > huge

def gradientdecent(model, data, targets, lossname, validata=None, valitargets=None, maxepoch=1E6, earlystop=False):
    """train fnn model by gradient decent

        Args:
            model:
            data:
            targets:
            lossname:
            validata: data used to calculate out-sample error
            valitargets: targets used to calculate out-sample error
            maxiteration: hard limit of learning iterations default is 10000
        Returns: final predictions and cost. Training will modify model.
    """

    import numpy as np
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.backprop import backprop
    from crpm.ffn_bodyplan import reinit_ffn

    #convergence test constants
    alpha_norm = 5E-5 #scales learning rate by max force relative to weight
    nbuffer = 500
    maxslope = -1E-6 #max learning slope should be negative but close to zero
    tgrid = np.array(range(nbuffer))
    tsum = np.sum(tgrid)
    tvar = nbuffer*np.sum(np.multiply(tgrid, tgrid))-tsum*tsum

    #bad starting point: reinitialize model if it has bad forces
    while model_has_bad_forces(model, data, targets, lossname):
        print("Runtime Warning in gradiendecent.py - reinitializing model")
        model = reinit_ffn(model)

    #calculate initial forces
    pred, state = fwdprop(data, model)
    cost, dloss = loss(lossname, pred, targets)
    forces = backprop(model, state, dloss)

    #check if using validation set
    is_validating = not ((validata is None) or (valitargets is None))
    #if so - calculate starting out-sample error
    if is_validating:
        pred, state = fwdprop(validata, model)
        cost, dloss = loss(lossname, pred, valitargets)

    #init best error and model
    best_cost = cost
    best_model = model

    #iterate training until:
    # 1) cost converges - defined as when slope of costbuffer is greater than to -1e-6
    # or
    # 2) out-sample error increases
    # or
    # 3) cost diverges - defined true when cost > 1E16
    # or
    # 4) too many iterations - hardcoded to ensure loop exit
    count = 0
    continuelearning = True
    while continuelearning:

        #clear cost buffer
        costbuffer = []

        #normalize learning rate alpha based on current forces
        maxf = []
        maxw = []
        for layer in forces:
            index = layer["layer"]
            maxf.append(np.max(abs(layer["fweight"])))
            maxw.append(np.max(abs(model[index]["weight"])))
        alpha = alpha_norm* np.nanmax(np.abs(np.divide(maxw, maxf)))

        #loop for training steps in buffer
        for i in tgrid:

            #update current learning step
            count += 1

            #update model
            for layer in forces:
                index = layer["layer"]
                model[index]["weight"] = model[index]["weight"] + alpha * layer["fweight"]
                model[index]["bias"] = model[index]["bias"] + alpha * layer["fbias"]

            #compute forces
            pred, state = fwdprop(data, model)
            cost, dloss = loss(lossname, pred, targets)
            forces = backprop(model, state, dloss)

            #record cost
            costbuffer.append(cost)

        #calculate cost slope to check for convergence
        slope = nbuffer*np.sum(np.multiply(tgrid, costbuffer))-tsum*np.sum(costbuffer)
        slope = slope/tvar

        #calculate out-sample error
        if is_validating:
            pred, _ = fwdprop(validata, model)
            cost, _ = loss(lossname, pred, valitargets)

        #Record best error and save model
        if cost <= best_cost:
            best_cost = cost
            best_model = model

        # - EXIT CONDITIONS -
        #exit if learning is taking too long
        if count > int(maxepoch):
            print("Warning gradientdecent.py: Training is taking a long time! - Try increaseing maxepoch - Training will end")
            continuelearning = False
        #exit if learning has plateaued
        if slope > maxslope:
            continuelearning = False
        #exit if early stopping and error has risen
        if  earlystop and cost > best_cost:
            print("early stopping")
            continuelearning = False
        #exit if cost has diverged
        if cost > 1E16:
            print("Warning gradientdecent.py: diverging cost function - try lowering learning rate or inc regularization constant - training will end.")
            continuelearning = False

    #return best model
    model = best_model

    #calculate final predictions and cost on best model
    if is_validating:
        pred, _ = fwdprop(validata, model)
        cost, _ = loss(lossname, pred, valitargets)
    else:
        pred, _ = fwdprop(data, model)
        cost, _ = loss(lossname, pred, targets)

    #return predictions and cost
    return pred, cost
