""" Compute state of ffn with from input dataset
"""

from crpm.activationfunctions import activation

def fwdprop(data, model):
    """ Compute network activity stemming from input dataset

    Args:
        data: An ndarray representing the vectorized values of M input samples
            arranged in columns with Nx features arranged in rows.
        model: A list of layer parameters represetning the model itself.
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

    #init state variables for input layer with input data
    state = []
    state.append(
        {
            "layer":0,
            "activity":data
        }
    )

    #calculate state variables for hidden layers
    prevlayeractivity = state[0]["activity"]
    for layer in model[1:]:
        stimulus = layer["weight"].dot(prevlayeractivity) + layer["bias"]
        activity = activation(layer["activation"], stimulus)
        prevlayeractivity = activity
        state.append(
            {
                "layer":layer["layer"],
                "activity":activity,
                "stimulus":stimulus
            }
        )

    #define prediction as top layer activity
    prediction = state[-1]["activity"]

    return prediction, state
