""" NN training by gradient decent
"""

def gradientdecent(model, data, targets, lossname, validata=None, valitargets=None):
    """train fnn model by gradient decent

        Args:
            model:
            data:
            targets:
            lossname:
            validata: data used to calculate out-sample error
            valitargets: targets used to calculate out-sample error
        Returns: final predictions and cost. Training will modify model.
    """

    import numpy as np
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.backprop import backprop

    #convergence test constants
    niter = 100
    tgrid = np.array(range(niter))
    tsum = np.sum(tgrid)
    tvar = niter*np.sum(np.multiply(tgrid, tgrid))-tsum*tsum


    #bad starting point: reinitialize model if it has bad forces
    while True:
        #do one fwd-back propagation pass
        pred, state = fwdprop(data, model)
        cost, dloss = loss(lossname, pred, targets)
        forces = backprop(model, state, dloss)

        #check for bad forces
        norm = np.linalg.norm(forces[-1]["fweight"])

        #exit condition
        if norm > 1e-16:
            break
        else:
            #reint model
            for i in range(1, len(model)):
                #print(i)
                ncurr = model[i]["weight"].shape[0]
                nprev = model[i]["weight"].shape[1]
                model[i]["weight"] = np.random.randn(ncurr, nprev)
                model[i]["bias"] = np.zeros((ncurr, 1))
            #print(norm,cost)

    #check if using validation set
    novalidation = (validata is None) or (valitargets is None)
    #if so - calculate starting out-sample error (ose)
    if novalidation:
        ose = cost
    else:
        pred, state = fwdprop(validata, model)
        ose, dloss = loss(lossname, pred, valitargets)
    #calculate best out-sample error
    best_ose = ose
    best_model = model

    #iterate training until cost converges - defined as when slope of costbuffer
    #is less than 1e-8
    while True:
        #clear cost buffer
        costbuffer = []

        #loop 100 training steps
        for i in tgrid:

            #normalize forces through learning rate alpha
            #mean abs(toplayer forces) < 0.0001 mean toplayer weights
            norm = np.linalg.norm(forces[-1]["fweight"])
            alpha = 0.0001 *  np.linalg.norm(model[-1]["weight"])/norm

            #update model
            for layer in forces:
                index = layer["layer"]
                model[index]["weight"] = model[index]["weight"] + alpha * layer["fweight"]
                model[index]["bias"] = model[index]["bias"] + alpha * layer["fbias"]

            #do one fwd-back propagation pass
            pred, state = fwdprop(data, model)
            cost, dloss = loss(lossname, pred, targets)
            forces = backprop(model, state, dloss)

            #record cost
            costbuffer.append(cost)

        #calculate out-sample error (ose)
        if novalidation:
            ose = cost
        else:
            pred, state = fwdprop(validata, model)
            ose, dloss = loss(lossname, pred, valitargets)
        #Record best out-sample error and save model
        if ose < best_ose:
            best_ose = ose
            best_model = model

        #calculate cost slope to check for convergence
        slope = niter*np.sum(np.multiply(tgrid, costbuffer))-tsum*np.sum(costbuffer)
        slope = slope/tvar

        #exit condition
        if abs(slope) <= 1E-8:
            break

    #get best model if validating otherwize use model at end of convergence
    if novalidation:
        model = best_model

    #calculate final predictions and cost
    if novalidation:
        pred, __ = fwdprop(data, model)
        cost, __ = loss(lossname, pred, targets)
    else:
        pred, __ = fwdprop(validata, model)
        cost, __ = loss(lossname, pred, valitargets)


    #return predictions and cost
    return pred, cost
