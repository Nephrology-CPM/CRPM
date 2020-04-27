""" dynamics functions for model as FFN object or as body of FFN class
"""

def computeforces(model, data, targets, lossname):
    """compute forces on weights and biases
    """
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.backprop import backprop
    from crpm.ffn import FFN

    if isinstance(model, FFN):
        pred = data
        #pre fwd prop if any
        if model.pre is not None:
            pred, _ = fwdprop(pred, model.pre)

        #body fwd prop
        pred, state = fwdprop(pred, model.body)

        #post fwd prop if any
        if model.post is not None:
            pred, poststate = fwdprop(pred, model.post)

        #get derivative of loss function
        _, dloss = loss(lossname, pred, targets)

        #post back prop if any
        if model.post is not None:
            _, dloss = backprop(model.post, poststate, dloss)

        #body back prop to get forces
        forces, _ = backprop(model.body, state, dloss)

        #return FFN forces
        return forces

    #If not FFN object then simply return body forces
    pred, state = fwdprop(data, model)
    _, dloss = loss(lossname, pred, targets)
    forces, _ = backprop(model, state, dloss)
    return forces

def computecost(model, data, targets, lossname):
    """compute predictions and cost
    """
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss
    from crpm.ffn import FFN

    #get predictions
    if isinstance(model, FFN):
        pred = data
        #pre fwd prop if any
        if model.pre is not None:
            pred, _ = fwdprop(pred, model.pre)

        #body fwd prop
        pred, _ = fwdprop(pred, model.body)

        #post fwd prop if any
        if model.post is not None:
            pred, _ = fwdprop(pred, model.post)

    #ELSE IF model is not FFN object then simply fwd prop to get predictions
    else:
        pred, _ = fwdprop(data, model)

    #calculate cost based on predictions
    cost, _ = loss(lossname, pred, targets)

    #return predictions and cost
    return pred, cost

def maxforce(model, forces):
    """ find max force on weights """
    import numpy as np
    from crpm.ffn import FFN

    # get FFN body if input is FFN object otherwise assume input is a body"""
    body = model
    if isinstance(model, FFN):
        body = model.body

    #init max force per layer array
    maxf = []
    for layer in forces:
        index = layer["layer"]
        if np.all(abs(body[index]["weight"]) >= np.finfo(float).eps):
            maxf.append(np.max(np.abs(np.divide(layer["fweight"],
                                            body[index]["weight"]))))

    return np.max(maxf)

def setupdynamics(model, data, targets, lossname):
    """ checks model for dynamics simulation
    """
    import numpy as np
    from crpm.ffn import FFN
    #check for huge forces relative to its respective weight
    huge = 1E16
    norm = huge
    attempt = 0
    print("setting up dynamics!")
    while norm >= huge and attempt < 100:
        attempt += 1
        # calculate forces
        forces = computeforces(model, data, targets, lossname)
        # get maxforce
        norm = maxforce(model, forces)
        #check for bad starting point: reinitialize model if it has bad forces
        if norm >= huge:
            #check if model is FFN object or FFN body
            if isinstance(model, FFN):
                model.reinit()
            else:
                model = reinit_ffn(model)
    if attempt >= 100:
        print("Error in setupdynamics.py - cannot reinitialize model ")
        return None
    print("Setup complete - initial forces calculated.")
    #return initial forces
    return forces
