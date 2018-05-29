""" Compute state of ffn with from input dataset
"""

def fwdprop(data, model):
    """ Compute network activity stemming from input dataset

    Args:
        data: An ndarray representing the vectorized values of M input samples
            arranged in columns with Nx features arranged in rows.
        model: A list of layer parameters represetning the model itself.
            Each layer is a list with keys and shapes "weight":(n,nprev), and
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

    prediction = []
    #L E  F T   O F F   H E R E
    prediction = data[0,]
    state = []

    return prediction, state
