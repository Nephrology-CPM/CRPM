""" dynamics functions
"""

def computeforces(model, data, targets, lossname):
    """compute forces on weights and biases
    """
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.backprop import backprop

    pred, state = fwdprop(data, model)
    _, dloss = loss(lossname, pred, targets)
    return backprop(model, state, dloss)

def computecost(model, data, targets, lossname):
    """compute predictions and cost
    """
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    pred, _ = fwdprop(data, model)
    cost, _ = loss(lossname, pred, targets)
    return pred, cost

def setupdynamics(model, data, targets, lossname):
    """ checks model for dynamics simulation
    """
    import numpy as np
    from crpm.ffn_bodyplan import reinit_ffn
    #check for huge forces relative to its respective weight
    huge = 1E16
    norm = huge
    attempt = 0
    while norm >= huge and attempt < 100:
        attempt += 1
        #calculate forces
        forces = computeforces(model, data, targets, lossname)
        maxf = []
        for layer in forces:
            index = layer["layer"]
            maxf.append(np.max(np.abs(np.divide(layer["fweight"],
                                                model[index]["weight"]))))
        norm = np.max(maxf)
        #check for bad starting point: reinitialize model if it has bad forces
        if norm >= huge:
            model = reinit_ffn(model)
    if attempt >= 100:
        print("Error in setupdynamics.py - cannot reinitialize model ")
        return None
    #return initial forces
    return forces

def normalizelearningrate(model, forces, alpha_norm):
    """ normalize learning rate alpha based on current forces
    """
    import numpy as np
    maxf = []
    maxw = []
    for layer in forces:
        index = layer["layer"]
        maxf.append(np.max(abs(layer["fweight"])))
        maxw.append(np.max(abs(model[index]["weight"])))
    alpha = alpha_norm* np.nanmax(np.abs(np.divide(maxw, maxf)))
    return alpha
