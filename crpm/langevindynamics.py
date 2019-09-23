""" NN training by langevin dynamics
"""

def langevindynamics(model, data, targets, lossname, validata=None,
                     valitargets=None, maxepoch=int(1E6), maxbuffer=int(1E3)):
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
    from crpm.dynamics import setupdynamics
    from crpm.dynamics import normalizelearningrate
    from crpm.dynamics import computecost
    from crpm.dynamics import computeforces
    from crpm.ffn_bodyplan import copy_ffn
    from crpm.pvalue import righttailpvalue

    #convergence test constants
    alpha_norm = 5E-5 #scales learning rate by max force relative to weight
    #nbuffer = 500
    nbuffer = 0
    #maxslope = -1E-6 #max learning slope should be negative but close to zero

    #buffer time grid
    #tgrid = np.array(range(nbuffer))
    #tsum = np.sum(tgrid)
    #tvar = nbuffer*np.sum(np.multiply(tgrid, tgrid))-tsum*tsum

    #langevin hyper parameters
    #eta = 5E-1 #ideal fraction of unexplained variance in costbuffer
    #downgamma = 0.95 #fraction by which friction is decreased
    #upgamma = 1.05 #fraction by which friction is decreased
    downtemp = 0.95 #fraction by which temperature is decreased
    uptemp = 1.05 #fraction by which temperature is increased

    #init lagevin parameters
    gamma = 5E-2 #viscosity or friction
    invbeta = 1E-6 #temperature ~ 1/beta

    #setup dynamics
    forces = setupdynamics(model, data, targets, lossname)

    #check if using validation set
    is_validating = not ((validata is None) or (valitargets is None))

    #define out-sample error calculator
    def out_sample_error():
        if is_validating:
            pred, cost = computecost(model, validata, valitargets, lossname)
        else:
            pred, cost = computecost(model, data, targets, lossname)
        return pred, cost

    #calculate out-sample error
    _, cost = out_sample_error()

    #init best error and model
    best_cost = copy.copy(cost)
    best_model = copy_ffn(model)

    #init cost history
    costhistory = np.full(maxbuffer, cost)

    #iterate training until:
    # 1) cost diverges - defined true when cost > 1E16
    # or
    # 2) too many iterations - hardcoded to ensure loop exit
    epoch = 0
    window = 0
    continuelearning = True
    while continuelearning:

        ##clear cost buffer
        #costbuffer = []

        #save cost at begining of buffer
        init_cost = copy.copy(cost)

        #normalize learning rate alpha based on current forces
        alpha = normalizelearningrate(model, forces, alpha_norm)

        #calculate langevin dynamics factors
        timestep = np.sqrt(2*alpha)
        halftimestep = timestep/2
        littled = np.exp(-gamma*timestep)
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
                model[index]["weight"] = (model[index]["weight"] +
                                          halftimestep * model[index]["weightdot"])
                model[index]["bias"] = (model[index]["bias"] +
                                        halftimestep * model[index]["biasdot"])

            #compute forces
            forces = computeforces(model, data, targets, lossname)

            #update model momenta by whole step
            for layer in forces:
                index = layer["layer"]
                ncurr = model[index]["n"]
                nprev = model[index-1]["n"]
                model[index]["weightdot"] = (littled*model[index]["weightdot"] +
                                             littleq * layer["fweight"] +
                                             sigma *  np.random.randn(ncurr, nprev))
                model[index]["biasdot"] = (littled*model[index]["biasdot"] +
                                           littleq * layer["fbias"] +
                                           sigma *  np.random.randn(ncurr, 1))

            #update model postions by second half-step
            for layer in forces:
                index = layer["layer"]
                model[index]["weight"] = (model[index]["weight"] +
                                          halftimestep * model[index]["weightdot"])
                model[index]["bias"] = (model[index]["bias"] +
                                        halftimestep * model[index]["biasdot"])

            ##record cost at full step
            #costbuffer.append(computecost(model, data, targets, lossname))

        #calculate out-sample error
        _, cost = out_sample_error()

        #increment window counter and save out sample error in cost history
        window += 1
        costhistory[window%maxbuffer] = copy.copy(cost)

        #Record best error and save model
        if cost <= best_cost:
            best_cost = copy.copy(cost)
            best_model = copy_ffn(model)

        #linear regression and goodness of fit measures in buffer
        #ysum = np.sum(costbuffer) # sum of costbuffer
        #in-sample error slope
        #slope = (nbuffer*np.sum(np.multiply(tgrid, costbuffer))-tsum*ysum)/tvar
        #ntercept = (ysum-slope*tsum)/nbuffer #in-sample error y-intercept
        #residuals = np.subtract(costbuffer,(slope*tgrid+intercept)) #fit error
        ##explained error sum of squares times nbuffer
        #sserr = nbuffer*np.sum(np.multiply(residuals,residuals))
        ##total error sum of squares times nbuffer
        #sstot = nbuffer*np.sum(np.multiply(costbuffer, costbuffer))-ysum*ysum
        #fvu = sserr/sstot #fraction of variance unexplained
        out_slope = (cost-init_cost)#/nbuffer #out-sample(validation) slope

        #Thermostat
        #if out_slope is negative
        #then decrease temperature
        #else increase temperature with probability p_out
        #where p_out is the proportion of out sample error historical values
        #that are greater than the current out sample error
        #in other words p_out is the right-tailed p_value of the out sample error.
        if out_slope < 0:
            invbeta *= downtemp
            #print(" ")
            #print("- temp "+str(invbeta))
        else:
            pvalue = righttailpvalue(np.array([cost]), costhistory)
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
        #    print("temp = "+str(invbeta)+"    KE = "+str(keng)+"    <cost> = "
        #          +str(np.mean(costhistory))+"    cost = "+str(cost)+
        #          "    best cost = "+str(best_cost))

        # - EXIT CONDITIONS -
        #exit if learning is taking too long
        if epoch > maxepoch:
            print("Warning langevindynamics.py: Training is taking a long time!"+
                  " - Try increaseing maxepoch - Training will end")
            continuelearning = False
        #exit if cost has diverged
        if cost > 1E16:
            print("Warning langevindynamics.py: diverging cost function "+
                  "- try lowering learning rate or inc regularization constant"+
                  " - training will end.")
            continuelearning = False
            #model = copy_ffn(best_model)

    #return best model
    model = copy_ffn(best_model)

    #return predictions and cost
    return  out_sample_error()
