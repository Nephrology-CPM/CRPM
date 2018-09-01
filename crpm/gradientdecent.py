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
    norm = 0
    #while True:
    while norm < 1e-16 or norm > 1e16:
        #do one fwd-back propagation pass
        pred, state = fwdprop(data, model)
        cost, dloss = loss(lossname, pred, targets)
        forces = backprop(model, state, dloss)

        #check for bad forces
        #norm = np.linalg.norm(forces[-1]["fweight"])
        maxf = []
        maxw = []
        for layer in forces:
            index = layer["layer"]
            maxf.append(np.max(abs(layer["fweight"])))
            maxw.append(np.max(abs(model[index]["weight"])))
        norm = np.max(np.abs(np.divide(maxf, maxw)))

        #exit condition
        if norm < 1e-16 or norm > 1e16:
            #reint model
            for i in range(1, len(model)):
                #print(i)
                ncurr = model[i]["weight"].shape[0]
                nprev = model[i]["weight"].shape[1]
                model[i]["weight"] = np.random.randn(ncurr, nprev)
                model[i]["bias"] = np.zeros((ncurr, 1))
            print("Runtime Warning in gradiendecent.py - reinitializing model",
                  "max weight= ", np.max(maxw), ", max force= ", np.max(maxf))


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
    #is less than or equal to 1e-8
    slope = 1
    count = 0
    while abs(slope) > 1E-6:
    #while abs(slope) > 1E-7:
    #while abs(slope) > 1E-8:
    #while abs(slope) > 1E-9:
    #while abs(slope) > 1E-10:
    #while True:

        #warn if loop is taking too long
        count += 1
        if count > 10000:
            print("Warning gradientdecent.py: Training is taking a long time! - training will end")
            break

        #clear cost buffer
        costbuffer = []

        #normalize forces through learning rate alpha
        #max(forces) < 0.0001 max(weights)
        maxf = []
        maxw = []
        for layer in forces:
            index = layer["layer"]
            maxf.append(np.max(abs(layer["fweight"])))
            maxw.append(np.max(abs(model[index]["weight"])))
        alpha = 0.0001* np.nanmax(np.abs(np.divide(maxw, maxf)))
        #alpha = 0.0001 *  np.max(maxw)/np.max(maxf)

        #loop 100 training steps
        for i in tgrid:

            #normalize forces through learning rate alpha
            #mean abs(toplayer forces) < 0.0001 mean toplayer weights
            #norm = np.linalg.norm(forces[-1]["fweight"])
            #alpha = 0.0001 *  np.linalg.norm(model[-1]["weight"])/norm

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
        if ose <= best_ose:
            best_ose = ose
            best_model = model

        #calculate cost slope to check for convergence
        slope = niter*np.sum(np.multiply(tgrid, costbuffer))-tsum*np.sum(costbuffer)
        slope = slope/tvar

        #exit condition
        #if abs(slope) <= 1E-8:
        #    break

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
