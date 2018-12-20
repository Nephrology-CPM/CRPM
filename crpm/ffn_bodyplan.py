"""General feed forward network body plan.
"""

def read_bodyplan(file):
    """Read body plan from csv file.

    Args:
        file: path to file containing a bodyplan
    Returns:
         A list of L layers representing the model structure with layer
         keys "layer", "n", "activation", and "regval" containing the layer index,
         integer number of units in that layer, the name of the
         activation function employed respectively, and the value of L2
         regularization to employ.
    """
    #init bodyplan as empty list
    bodyplan = []

    import csv
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        keys = list(next(reader)) #get headers
        for line in reader:
            layer = {}
            layer["layer"] = int(line[keys.index("layer")])
            layer["n"] = int(line[keys.index("n")])
            layer["activation"] = str(line[keys.index("activation")])
            if "regval" in keys:
                layer["regval"] = float(line[keys.index("regval")])
            else:
                layer["regval"] = float(0)
            bodyplan.append(layer)
    return bodyplan

def init_ffn(bodyplan):
    """Setup for arbitrary feed forward network model.

    Args:
        bodyplan: A list of L layers representing the model architecture with layers
            represented as dictionaries with keys "layer", "n", "activation",
            and "regval". These keys contain the layer index,
            integer number of units in that layer, the name of the
            activation function employed, and the L2 regularization parameter employed respectively.
    Returns:
        A list of layers with parameters to be optimized by the learning algorithm
        represetning the current model state.
        Each layer is a dictionary with keys and shapes "weight":(n,nprev), and
        "bias" (n, 1) so function returns a list of dicttionaries.
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
            "regval":bodyplan[layer]["regval"],
            "weight": np.random.randn(ncurr, nprev), #random initial weights
            "bias": np.zeros((ncurr, 1)) # zeros for initial biases
            })

    return model
