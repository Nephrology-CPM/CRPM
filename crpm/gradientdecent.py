""" NN training by gradient decent
"""

def gradientdecent(model, data, targets, lossname, validata=None,
                   valitargets=None, maxepoch=1E6, earlystop=False):
    """train fnn model by gradient decent

        Args:
            model:
            data:
            targets:
            lossname:
            validata: data used to calculate out-sample error
            valitargets: targets used to calculate out-sample error
            maxiteration: hard limit of learning iterations default is 10000
        Returns: final predictions and cost along with exit condition.
            Exit conditions are 0) learning converged, 1) learning not
            converged, 2) learning was stopped early, and -1) learning diverged.
            Training will modify model.
    """

    import numpy as np
    from crpm.dynamics import setupdynamics
    from crpm.dynamics import normalizelearningrate
    from crpm.dynamics import computecost
    from crpm.dynamics import computeforces
    from crpm.ffn_bodyplan import copy_ffn

    #convergence test constants
    alpha_norm = 5E-5 #scales learning rate by max force relative to weight
    nbuffer = 500
    maxslope = -1E-6 #max learning slope should be negative but close to zero
    tgrid = np.array(range(nbuffer))
    tsum = np.sum(tgrid)
    tvar = nbuffer*np.sum(np.multiply(tgrid, tgrid))-tsum*tsum

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
    best_cost = np.copy(cost)
    best_model = copy_ffn(model)

    #iterate training until:
    # 1) cost converges - defined as when slope of costbuffer is greater than to -1e-6
    # or
    # 2) out-sample error increases
    # or
    # 3) cost diverges - defined true when cost > 1E16
    # or
    # 4) too many iterations - hardcoded to ensure loop exit
    continuelearning = True
    #Do not do any learning if maxepoch is not a positive integer
    if maxepoch<1 :
        continuelearning = False
    count = 0
    exitcond = 0
    while continuelearning:

        #clear cost buffer
        costbuffer = []

        #normalize learning rate alpha based on current forces
        alpha = normalizelearningrate(model, forces, alpha_norm)

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
            forces = computeforces(model, data, targets, lossname)

            #record cost
            _, cost = computecost(model, data, targets, lossname)
            costbuffer.append(cost)

        #calculate cost slope to check for convergence
        slope = nbuffer*np.sum(np.multiply(tgrid, costbuffer))-tsum*np.sum(costbuffer)
        slope = slope/tvar

        #calculate out-sample error
        _, cost = out_sample_error()

        #Record best error and save model
        if cost <= best_cost:
            #print(cost)
            best_cost = np.copy(cost)
            best_model = copy_ffn(model)

        # - EXIT CONDITIONS -
        #exit if learning is taking too long
        if count > int(maxepoch):
            print("Warning gradientdecent.py: Training is taking a long time!"+
                  " - Try increaseing maxepoch - Training will end")
            exitcond = 1
            continuelearning = False
        #exit if learning has plateaued
        if slope > maxslope:
            exitcond = 0
            continuelearning = False
        #exit if early stopping and error has risen
        if  earlystop and cost > best_cost:
            print("early stopping")
            exitcond = 2
            continuelearning = False
        #exit if cost has diverged
        if cost > 1E16:
            print("Warning gradientdecent.py: diverging cost function "+
                  "- try lowering learning rate or inc regularization constant "+
                  "- training will end.")
            exitcond = -1
            continuelearning = False

    #return best model
    model = copy_ffn(best_model)

    #return predictions and cost
    return (*out_sample_error(),exitcond)
