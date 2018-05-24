"""General deep neural network body plan.
"""

def read_bodyplan(file):
    """Read body plan from csv file.

    Args:
        file: path to file containing a bodyplan
    Returns:
        A data frame representing the model structure with
        L layers arranged in rows with column variables "layer", "n" and "activation"
        containing respectively the layer index, integer number of units in
        that layer, and the name of the activation function employed.
    """
    import pandas as pd
    body_plan = pd.read_csv(file)

    return body_plan
