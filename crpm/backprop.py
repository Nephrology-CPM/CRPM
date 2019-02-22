"""Compute forces on model based on loss function
"""

import numpy as np
from crpm.activationfunctions import dactivation

def model_has_bad_forces(model, data, targets, lossname):
    """ Does input data result in model forces that break integrator?"""
    from crpm.fwdprop import fwdprop
    from crpm.lossfunctions import loss

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
        maxf.append(np.max(np.abs(np.divide(layer["fweight"],model[index]["weight"]))))
    norm = np.max(maxf)

    #return True if forces are huge
    return norm > huge


def backprop(model, state, dloss):
    """ Compute network forces based on activity and loss metric

    Args:
        model: A list of layer parameters represetning the model itself.
            Each layer is a dict with keys and shapes "weight":(n,nprev), and
            "bias" (n, 1).
        state: A list of layer activities and stimuli representing the
            current model state cached for use in the subsequent backward
            propagation step of the learning algorithm with keys and shapes
            [{"activity":(Nx, M)}
           lossctivity":(N1, M), "stimulus":(N1, M)},
            {"activity":(N2, M), "stimulus":(N2, M)},
            ...,
            {"activity":(NL, M), "stimulus":(NL, M)}].
        dloss: array of derivaties of loss function with respect to top layer
            activity

    Returns:
        forces: A list of layer parameter forces.
        Each layer is a dict with keys and shapes "fweight":(n,nprev), and
        "fbias" (n, 1).
    """

    #init forces
    forces = []

    #init top layer derivative w.r.t. activity with dloss
    dact = dloss

    #store number of samples
    norm = model[-1]["bias"].shape[1]

    #loop over layers (top to bottom) - calculatate dZ, dW, and db
    for layer in reversed(model[1:]):
        index = layer["layer"]

        #calculate layer derivative w.r.t. stimulus using dact of layer above
        dstim = dact * dactivation(layer["activation"], state[index]["stimulus"])
        #calculate layer derivative w.r.t. weight using dstim
        dweight = dstim.dot(state[index-1]["activity"].T)/norm
        #add regularization term if specified by layer
        if layer["regval"] > 0:
            if layer["lreg"] == 1:
                dweight += layer["regval"]*np.sign(layer["weight"])
            elif layer["lreg"] == 2:
                dweight += layer["regval"]*layer["weight"]
        #calculate layer derivative w.r.t. bias using dstim
        dbias = np.sum(dstim, axis=1, keepdims=True)/norm
        #calculate next layer down deriv wrt activity
        dact = layer["weight"].T.dot(dstim)
        # add forces to begining of list
        forces.insert(0,
                      {
                          "layer":layer["layer"],
                          "fweight":-dweight,
                          "fbias":-dbias
                      }
                     )

    return forces
