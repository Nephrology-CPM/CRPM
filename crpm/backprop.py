"""FNN backpropagation scheme

TODO: have independent regularization terms for weights and biases

"""


def backprop(body, state, dloss):
    """ Compute network forces based on activity and loss metric

    Args:
        model: FFN object or FFN body as a list of layer parameters representing
            the model itself. Each layer is a dict with keys and shapes
            "weight":(n,nprev), and "bias" (n, 1).
        state: A list of layer activities and stimuli representing the
            current model state with keys and shapes
            [{"activity":(Nx, M)}
           lossctivity":(N1, M), "stimulus":(N1, M)},
            {"activity":(N2, M), "stimulus":(N2, M)},
            ...,
            {"activity":(NL, M), "stimulus":(NL, M)}].
        dloss: array of derivaties of loss function with respect to top layer
            activity.
            If model is an FFN object then the dloss is calcualted on the
            post-processed top layer activity.

    Returns:
        forces: A list of layer parameter forces.
        Each layer is a dict with keys and shapes "fweight":(n,nprev), and
        "fbias" (n, 1).
    """
    import numpy as np
    from crpm.activationfunctions import dactivation

    #only works on FFN.bodies not on FFN objects
    #check if model is FFN object or FFN body
    #body = ffn_body(model)

    #init forces
    forces = []

    #init top layer derivative w.r.t. activity with dloss
    dact = dloss

    #store number of samples
    norm = body[-1]["bias"].shape[1]

    #loop over layers (top to bottom) - calculatate dZ, dW, and db
    for layer in reversed(body[1:]):
        index = layer["layer"]

        #calculate layer derivative w.r.t. stimulus using dact of layer above
        dstim = dact * dactivation(layer["activation"], state[index]["stimulus"])
        #calculate layer derivative w.r.t. weight using dstim
        dweight = dstim.dot(state[index-1]["activity"].T)/norm
        #calculate layer derivative w.r.t. bias using dstim
        dbias = np.sum(dstim, axis=1, keepdims=True)/norm
        #add regularization term if specified by layer
        if layer["regval"] > 0:
            if layer["lreg"] == 1:
                dweight += layer["regval"]*np.sign(layer["weight"])
                dbias += layer["regval"]*np.sign(layer["bias"])
            elif layer["lreg"] == 2:
                dweight += layer["regval"]*layer["weight"]
                dbias += layer["regval"]*layer["bias"]
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
    return forces, dact
