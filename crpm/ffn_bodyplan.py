"""General feed forward network body plan.
"""

#global parameter
init_weight_std = 1.0

def read_bodyplan(file):
    """Read body plan from csv file.

    Args:
        file: path to file containing a bodyplan
    Returns:
         A list of L layers representing the model structure with layer
         keys: "layer", "n", "activation", "lreg", "regval", and "desc"
         indicating the layer index, integer number of units in layer, a string
         name of the activation function, the value (1 or 2) of the L-norm
         regularization to employ, the float regularization constant, and a
         string description of layer.
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
                layer["regval"] = float(0) #default regularization parameter
            if "lreg" in keys:
                layer["lreg"] = int(line[keys.index("lreg")])
            else:
                layer["lreg"] = int(1) #default regularization is L1
            if "desc" in keys:
                layer["desc"] = str(line[keys.index("desc")])
            else:
                layer["desc"] = 'fully connected'
            bodyplan.append(layer)
    return bodyplan

def init_ffn(bodyplan, weightstd=None):
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
        "bias" (n, 1) so function returns a list of dictionaries.
    """
    import numpy as np

    #get random initial weight std
    if weightstd is None:
        weightstd = init_weight_std
    #print todo # WARNING:
    #print("WARNING need to initialize weights with small std ~.01")

    #init model as list holding data for each layer start with input layer
    model = []
    model.append({
                "layer":0,
                "n":bodyplan[0]['n'],
                "activation": bodyplan[0]["activation"],
                "lreg":bodyplan[0]["lreg"],
                "regval":bodyplan[0]["regval"],
                "desc": bodyplan[0]["desc"]
                })

    # init weights and biases for hidden layers and declare activation function
    for layer in range(1, len(bodyplan)):
        ncurr = bodyplan[layer]["n"]
        nprev = bodyplan[layer-1]["n"]
        model.append({
            "layer":layer,
            "n":bodyplan[layer]['n'],
            "activation": bodyplan[layer]["activation"],
            "lreg":bodyplan[layer]["lreg"],
            "regval":bodyplan[layer]["regval"],
            "desc": bodyplan[layer]["desc"],
            "weight": np.random.randn(ncurr, nprev)*weightstd, #random initial weights
            "bias": np.zeros((ncurr, 1)), # zeros for initial biases
            "weightdot": np.zeros((ncurr, nprev)), #zeros for initial weight momenta
            "biasdot": np.zeros((ncurr, 1)) # zeros for initial bias momenta
            })

    return model

def get_bodyplan(model):
    """get bodyplan for arbitrary feed forward network model.

    Args:
        An ffn model
    Returns:
        bodyplan: A list of L layers representing the model architecture with layers
            represented as dictionaries with keys "layer", "n", "activation",
            and "regval". These keys contain the layer index,
            integer number of units in that layer, the name of the
            activation function employed, and the L2 regularization parameter employed respectively.
    """
    import numpy as np

    #init bodyplan as empty list
    bodyplan = []

    for mlayer in model:
        layer = {}
        layer["layer"] = mlayer["layer"]
        layer["n"] = mlayer["n"]
        layer["activation"] = mlayer["activation"]
        if "regval" in mlayer:
            layer["regval"] = mlayer["regval"]
        else:
            layer["regval"] = float(0) #default regularization parameter
        if "lreg" in mlayer:
            layer["lreg"] = mlayer["lreg"]
        else:
            layer["lreg"] = int(1) #default regularization is L1
        if "desc" in mlayer:
            layer["desc"] = mlayer["desc"]
        else:
            layer["desc"] = 'fully connected' #default description
        bodyplan.append(layer)

    return bodyplan

def new_layer(layer=None, n=1, activation='linear', regval=float(0), lreg=int(1), desc='fully connected'):
    """specify a new bodyplan layer"""
    layer = {}
    layer["layer"] = layer
    layer["n"] = n
    layer["activation"] = activation
    layer["regval"] = regval
    layer["lreg"] = lreg
    layer["desc"] = desc
    return layer

def stack_new_layer(model, weightstd=None, n=1, activation='linear',
                    regval=float(0), lreg=int(1), desc='fully connected'):
    """will add a new specified layer to top layer of model"""

    import numpy as np

    #get random initial weight std
    if weightstd is None:
        weightstd = init_weight_std

    nprev = model[-1]["n"]
    model.append({
        "layer":len(model),
        "n":n,
        "activation": activation,
        "lreg": lreg,
        "regval": regval,
        "desc": desc,
        "weight": np.random.randn(n, nprev)*weightstd, #random initial weights
        "bias": np.zeros((n, 1)), # zeros for initial biases
        "weightdot": np.zeros((n, nprev)), #zeros for initial weight momenta
        "biasdot": np.zeros((n, 1)) # zeros for initial bias momenta
        })

    return model


def copy_bodyplan(bodyplan):
    """get bodyplan for arbitrary feed forward network model.

    Args:
        An ffn model
    Returns:
        bodyplan: A list of L layers representing the model architecture with layers
            represented as dictionaries with keys "layer", "n", "activation",
            and "regval". These keys contain the layer index,
            integer number of units in that layer, the name of the
            activation function employed, and the L2 regularization parameter employed respectively.
    """
    import numpy as np
    import copy

    #init bodyplan as empty list
    newbodyplan = []

    for mlayer in bodyplan:
        layer = {}
        layer["layer"] = copy.copy(mlayer["layer"])
        layer["n"] = copy.copy(mlayer["n"])
        layer["activation"] = copy.copy(mlayer["activation"])
        layer["regval"] = copy.copy(mlayer["regval"])
        layer["lreg"] = copy.copy(mlayer["lreg"])
        layer["desc"] = copy.copy(mlayer["desc"])
        newbodyplan.append(layer)

    return newbodyplan

def push_bodyplanlayer(bodyplan,layer):
    """stack a layer onto exisiting bodyplan for arbitrary feed forward network model.

    Args:
        bodyplan: An ffn bodyplan with L layers
        layer: a bodyplan layer to push
    Returns:
        None : will modify bodyplan inplace as a list of L+1 layers representing the model architecture with layers
            represented as dictionaries with keys "layer", "n", "activation",
            and "lreg", "regval". These keys contain the layer index,
            integer number of units in that layer, the name of the
            activation function employed, the L integer of Lnorm regularization
            employed, and the regularization constant respectively.
    """
    import numpy as np
    import copy

    #copy layer
    newlayer = {}
    newlayer["layer"] = copy.copy(layer["layer"])
    newlayer["n"] = copy.copy(layer["n"])
    newlayer["activation"] = copy.copy(layer["activation"])
    newlayer["regval"] = copy.copy(layer["regval"])
    newlayer["lreg"] = copy.copy(layer["lreg"])
    newlayer["desc"] = copy.copy(layer["desc"])

    #get bodyplan depth
    nlayer = len(bodyplan)

    #assign newlayer the correct layer index
    newlayer["layer"] = nlayer

    #push layer
    bodyplan.append(newlayer)

    return

def reinit_ffn(model, weightstd=None):
    """Reinitialize feed forward network model.

    Args:
        model: A previously created ffn model
    Returns:
        The input model with reinitialized weights and biases
    """
    import numpy as np

    #always inform user model is being reinitialized
    print("Reinitialing Model!")

    #get random initial weight std
    if weightstd is None:
        weightstd = init_weight_std

    #init model as list holding data for each layer start with input layer
    newmodel = []
    newmodel.append({
                "layer":0,
                "n":model[0]['n'],
                "activation": model[0]["activation"],
                "lreg": model[0]["lreg"],
                "regval": model[0]["regval"],
                "desc": model[0]["desc"]
                })

    # init weights and biases for hidden layers and declare activation function
    for layer in range(1, len(model)):
        ncurr = model[layer]["n"]
        nprev = model[layer-1]["n"]
        newmodel.append({
            "layer": layer,
            "n": model[layer]['n'],
            "activation": model[layer]["activation"],
            "lreg": model[layer]["lreg"],
            "regval": model[layer]["regval"],
            "desc": model[layer]["desc"],
            "weight": np.random.randn(ncurr, nprev)*weightstd, #random initial weights
            "bias": np.zeros((ncurr, 1)), # zeros for initial biases
            "weightdot": np.zeros((ncurr, nprev)), #zeros for initial weight momenta
            "biasdot": np.zeros((ncurr, 1)) # zeros for initial bias momenta
            })

    return newmodel

def copy_ffn(model):
    """Copy feed forward network model.

    Args:
        model: A previously created ffn model
    Returns:
        A copy of the model
    """
    import numpy as np
    import copy

    #init model as list holding data for each layer start with input layer
    newmodel = []
    newmodel.append({
                "layer":0,
                "n": copy.copy(model[0]['n']),
                "activation": copy.copy(model[0]["activation"]),
                "lreg": copy.copy(model[0]["lreg"]),
                "regval": copy.copy(model[0]["regval"]),
                "desc": copy.copy(model[0]["desc"])
                })

    # init weights and biases for hidden layers and declare activation function
    for layer in range(1, len(model)):
        newmodel.append({
            "layer":layer,
            "n": copy.copy(model[layer]['n']),
            "activation": copy.copy(model[layer]["activation"]),
            "lreg": copy.copy(model[layer]["lreg"]),
            "regval": copy.copy(model[layer]["regval"]),
            "desc": copy.copy(model[layer]["desc"]),
            "weight": np.copy(model[layer]["weight"]),
            "bias": np.copy(model[layer]["bias"]),
            "weightdot": np.copy(model[layer]["weightdot"]),
            "biasdot": np.copy(model[layer]["biasdot"])
            })
    return newmodel
