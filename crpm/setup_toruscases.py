""" set up logistic classification model for intorus dataset
"""

def setup_toruscases():
    """ will return model and downloaded data."""

    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset

    #create model from bodyplan file
    bodyplan = read_bodyplan("crpm/data/intorus_bodyplan.csv")

    #create model
    model = init_ffn(bodyplan)

    #download data
    __, data = load_dataset("crpm/data/intorus.csv")

    return model, data

def setup_toruscases_deep():
    """ will return model and downloaded data."""

    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset

    #create model from deep bodyplan file
    bodyplan = read_bodyplan("crpm/data/intorus_deep_bodyplan.csv")

    #create model
    model = init_ffn(bodyplan)

    #download data
    __, data = load_dataset("crpm/data/intorus.csv")

    return model, data
