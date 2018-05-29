"""General feed forward network body plan.
"""

def read_bodyplan(file):
    """Read body plan from csv file.

    Args:
        file: path to file containing a bodyplan
    Returns:
         A list of L layers representing the model structure with layer
         keys "layer", "n" and "activation" containing the layer index,
         integer number of units in that layer, and the name of the
         activation function employed respectively.
    """
    #init bodyplan as empty list
    bodyplan = []

    import csv
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        keys = list(next(reader)) #get headers
        for layer in reader:
            bodyplan.append({keys[0]:int(layer[0]),
                             keys[1]:int(layer[1]),
                             keys[2]:layer[2]})
    return bodyplan

def init_ffn(bodyplan):
    """Setup for arbitrary feed forward network model.

    Args:
        bodyplan: A list of L layers representing the model structure with layer
            keys "layer", "n" and "activation" containing the layer index,
            integer number of units in that layer, and the name of the
            activation function employed respectively.
    Returns:
        A list of layer parameters to be optimized by the learning algorithm
        represetning the model itself.
        Each layer is a list with keys and shapes "weight":(n,nprev), and
        "bias" (n, 1) so function returns a list of lists.
    """
    import numpy as np

    #init model as list holding data for each layer start with input layer
    model = []
    model.append({
        "layer":0,
        "n":bodyplan[0]['n'],
        "activation": bodyplan[0]["activation"]
        })

    # init weights and biases for hidden layers and declare activation function
    for layer in range(1, len(bodyplan)):
        ncurr = bodyplan[layer]["n"]
        nprev = bodyplan[layer-1]["n"]
        model.append({
            "layer":layer,
            "n":bodyplan[layer]['n'],
            "activation": bodyplan[layer]["activation"],
            "weight": np.random.randn(ncurr, nprev), #random initial weights
            "bias": np.zeros((ncurr, 1)) # zeros for initial biases
            })

    return model
