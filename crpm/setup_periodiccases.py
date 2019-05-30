""" set up logistic classification model for 1D periodiccases dataset
"""

def setup_periodiccases():
    """ will return model and downloaded data."""

    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset

    #create model from bodyplan file
    bodyplan = read_bodyplan("crpm/data/periodiccases_bodyplan.csv")

    #create model
    model = init_ffn(bodyplan)

    #download data
    __, data = load_dataset("crpm/data/periodiccases.csv")

    return model, data

def setup_periodiccases_deep():
    """ will return model and downloaded data."""

    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset

    #create model from deep bodyplan file
    bodyplan = read_bodyplan("crpm/data/periodiccases_deep_bodyplan.csv")

    #create model
    model = init_ffn(bodyplan)

    #download data
    __, data = load_dataset("crpm/data/periodiccases.csv")

    return model, data
