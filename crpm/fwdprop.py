""" Compute state of ffn with from input dataset
"""

def fwdprop(data, body):
    """ Compute network activity stemming from input dataset

    Args:
        data: An ndarray representing the vectorized values of M input samples
            arranged in columns with Nx features arranged in rows.
            If model is an FFN object then the data will be transformed by
            the FFN static pre-processing component.
        model: FFN object or FFN body as a list of layer parameters
            representing the model itself.
            Each layer is a dictionary with keys and shapes "weight":(n,nprev), and
            "bias" (n, 1).

    Returns:
        prediction: An ndarray of shape (NL, M) representing the activity of
            the top layer where NL is the number of units in the top layer.
        state: A list of layer activities and stimuli representing the
            current model state cached for use in the subsequent backward
            propagation step of the learning algorithm with keys and shapes
            [{"activity":(Nx, M)}
            {"activity":(N1, M), "stimulus":(N1, M)},
            {"activity":(N2, M), "stimulus":(N2, M)},
            ...,
            {"activity":(NL, M), "stimulus":(NL, M)}].
    """
    import numpy as np
    from crpm.activationfunctions import activation

    #init activity with data
    #activity = data

    #init state variables for input layer with input data
    state = []
    state.append(
        {
            "layer":0,
            "activity":data #activity
        }
    )

    #calculate state variables for hidden layers
    for layer in body[1:]:

        stimulus = layer["weight"].dot(state[-1]["activity"]) + layer["bias"]
        #stimulus = layer["weight"].dot(activity) + layer["bias"]
        activity = activation(layer["activation"], stimulus)
        state.append(
            {
                "layer":layer["layer"],
                "activity":activity,
                "stimulus":stimulus
            }
        )

    #return predictions as top layer activity and the model state
    return state[-1]["activity"], state
