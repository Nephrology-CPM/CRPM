""" NN training by langevin dynamics
"""

def langevindynamics(model, data, targets, lossname, validata=None, valitargets=None, maxepoch=int(1E6), maxbuffer=int(1E3)):
    """train fnn model by langevin dynamics

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
    import copy
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.backprop import backprop
    from crpm.backprop import model_has_bad_forces
    from crpm.ffn_bodyplan import reinit_ffn
    from crpm.ffn_bodyplan import copy_ffn
    from crpm.pvalue import righttailpvalue

    #convergence test constants
    alpha_norm = 5E-5 #scales learning rate by max force relative to weight
    nbuffer = 500
    #maxslope = -1E-6 #max learning slope should be negative but close to zero

    #buffer time grid
    tgrid = np.array(range(nbuffer))
    tsum = np.sum(tgrid)
    tvar = nbuffer*np.sum(np.multiply(tgrid, tgrid))-tsum*tsum

    #langevin hyper parameters
    #eta = 5E-1 #ideal fraction of unexplained variance in costbuffer
    #downgamma = 0.95 #fraction by which friction is decreased
    #upgamma = 1.05 #fraction by which friction is decreased
    downtemp = 0.95 #fraction by which temperature is decreased
    uptemp = 1.05 #fraction by which temperature is increased

    #init lagevin parameters
    gamma = 5E-2 #viscosity or friction
    invbeta = 1E-6 #temperature ~ 1/beta

    #bad starting point: reinitialize model if it has bad forces
    while model_has_bad_forces(model, data, targets, lossname):
        print("Runtime Warning in langevindynamics.py - reinitializing model")
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
    best_cost = copy.copy(cost)
    best_model = copy_ffn(model)

    #init cost history
    costhistory = np.full(maxbuffer,cost)

    #iterate training until:
    # 1) cost diverges - defined true when cost > 1E16
    # or
    # 2) too many iterations - hardcoded to ensure loop exit
    epoch = 0
    window = 0
    continuelearning = True
    while continuelearning:

        #clear cost buffer
        costbuffer = []

        #save cost at begining of buffer
        init_cost = copy.copy(cost)

        #normalize learning rate alpha based on current forces
        maxf = []
        maxw = []
        for layer in forces:
            index = layer["layer"]
            maxf.append(np.max(abs(layer["fweight"])))
            maxw.append(np.max(abs(model[index]["weight"])))
        alpha = alpha_norm* np.nanmax(np.abs(np.divide(maxw, maxf)))

        #calculate langevin dynamics factors
        dt = np.sqrt(2*alpha)
        halfdt = dt/2
        littled = np.exp(-gamma*dt)
        littleq = (1-littled)/gamma
        sigma = np.sqrt(invbeta*(1-gamma*gamma))

        #loop for training steps in buffer
        #for i in tgrid:
        for i in range(nbuffer):
            #update current learning step
            epoch += 1

            #update model postions by half step
            for layer in forces:
                index = layer["layer"]
                model[index]["weight"] = model[index]["weight"] + halfdt * model[index]["weightdot"]
                model[index]["bias"] = model[index]["bias"] + halfdt * model[index]["biasdot"]

            #compute forces
            pred, state = fwdprop(data, model)
            cost, dloss = loss(lossname, pred, targets)
            forces = backprop(model, state, dloss)

            #update model momenta by whole step
            for layer in forces:
                index = layer["layer"]
                ncurr = model[index]["n"]
                nprev = model[index-1]["n"]
                model[index]["weightdot"] = littled*model[index]["weightdot"] + littleq * layer["fweight"] + sigma *  np.random.randn(ncurr, nprev)
                model[index]["biasdot"] = littled*model[index]["biasdot"] + littleq * layer["fbias"] + sigma *  np.random.randn(ncurr, 1)

            #update model postions by second half-step
            for layer in forces:
                index = layer["layer"]
                model[index]["weight"] = model[index]["weight"] + halfdt * model[index]["weightdot"]
                model[index]["bias"] = model[index]["bias"] + halfdt * model[index]["biasdot"]

            #record cost at full step
            pred, _ = fwdprop(data, model)
            cost, _ = loss(lossname, pred, targets)
            costbuffer.append(cost)

        #calculate out-sample error
        if is_validating:
            pred, _ = fwdprop(validata, model)
            cost, _ = loss(lossname, pred, valitargets)

        #increment window counter and save in cost history
        window += 1
        costhistory[window%maxbuffer] = copy.copy(cost)

        #Record best error and save model
        if cost <= best_cost:
            best_cost = copy.copy(cost)
            best_model = copy_ffn(model)

        #linear regression and goodness of fit measures in buffer
        ysum = np.sum(costbuffer) # sum of costbuffer
        slope = (nbuffer*np.sum(np.multiply(tgrid, costbuffer))-tsum*ysum)/tvar #in-sample error slope
        #ntercept = (ysum-slope*tsum)/nbuffer #in-sample error y-intercept
        #residuals = np.subtract(costbuffer,(slope*tgrid+intercept)) #fit error
        #sserr = nbuffer*np.sum(np.multiply(residuals,residuals))#explained error sum of squares times nbuffer
        #sstot = nbuffer*np.sum(np.multiply(costbuffer, costbuffer))-ysum*ysum#total error sum of squares times nbuffer
        #fvu = sserr/sstot #fraction of variance unexplained
        out_slope = (cost-init_cost)#/nbuffer #out-sample(validation) slope

        #Thermostat
        #if out_slope is negative
        #then decrease temperature
        #else increase temperature with probability p_out
        #where p_out is the proportion of out sample error historical values that are greater than the current out sample error
        #in other words p_out is the right-tailed p_value of the out sample error.
        if out_slope < 0:
            invbeta *= downtemp
            #print(" ")
            #print("- temp "+str(invbeta))
        else:
            pvalue = righttailpvalue(np.array([cost]),costhistory)
            #print(" ")
            #print("pvalue = "+str(pvalue))
            if np.random.random() <= pvalue:
                #print("+ temp "+str(invbeta))
                invbeta *= uptemp

        #Viscostat
        #if fraction of unexplained variance is < eta
        #then decrease friction
        #else increase friction
        #where hyperparameter eta should be close to 0
        #if fvu < eta:
        #    gamma *= .95
        #else:
        #    gamma *= 1.05

        #if window%10==0:
        #    keng = 0
        #    for layer in model[1:]:
        #        keng += np.sum(np.multiply(layer["weightdot"],layer["weightdot"]))
        #    print("temp = "+str(invbeta)+"    KE = "+str(keng)+"    <cost> = "+str(np.mean(costhistory))+"    cost = "+str(cost)+"    best cost = "+str(best_cost))

        # - EXIT CONDITIONS -
        #exit if learning is taking too long
        if epoch > maxepoch:
            print("Warning langevindynamics.py: Training is taking a long time! - Try increaseing maxepoch - Training will end")
            continuelearning = False
        #exit if cost has diverged
        if cost > 1E16:
            print("Warning langevindynamics.py: diverging cost function - try lowering learning rate or inc regularization constant - training will end.")
            continuelearning = False
            #model = copy_ffn(best_model)

    #return best model
    model = copy_ffn(best_model)

    #calculate final predictions and cost on best model
    if is_validating:
        pred, _ = fwdprop(validata, model)
        cost, _ = loss(lossname, pred, valitargets)
    else:
        pred, _ = fwdprop(data, model)
        cost, _ = loss(lossname, pred, targets)

    #return predictions and cost
    return pred, cost
