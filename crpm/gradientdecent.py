""" NN training by gradient decent
"""

def gradientdecent(model, data, targets, lossname):
    """train fnn model by gradient decent

        Args:
            model:
            data:
            targets:
            lossname:
        Returns: final cost and will modify model
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

    #iterate training until cost converges - defined as when slope of costbuffer
    #is less than 1e-8
    while True:
        #clear cost buffer
        costbuffer = []

        #loop 100 training steps
        for i in tgrid:

            #do one fwd-back propagation pass
            pred, state = fwdprop(data, model)
            cost, dloss = loss(lossname, pred, targets)
            forces = backprop(model, state, dloss)

            #normalize forces through learning rate alpha
            #mean abs(toplayer forces) < 0.00001 mean toplayer weights
            alpha = 0.00001 *  np.linalg.norm(model[-1]["weight"])
            alpha /= np.linalg.norm(forces[-1]["dweight"])

            #record cost
            costbuffer.append(cost)

            #update model
            for layer in forces:
                index = layer["layer"]
                model[index]["weight"] = model[index]["weight"] + alpha * layer["dweight"]
                model[index]["bias"] = model[index]["bias"] + alpha * layer["dbias"]

        #calculate cost slope to check for convergence
        slope = niter*np.sum(np.multiply(tgrid, costbuffer))-tsum*np.sum(costbuffer)
        slope = slope/tvar

        print(slope, cost)

        #exit condition
        if abs(slope) <= 1E-10:
            break

    #return last cost
    return costbuffer[-1]
