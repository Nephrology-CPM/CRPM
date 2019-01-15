""" set up overfitting model
"""

def setup_overfitting_shallow():
    """ will return shallow model and downloaded data."""

    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset

    #create model from  bodyplan file
    bodyplan = read_bodyplan("crpm/data/overfitting_shallow_bodyplan.csv")

    #create model
    model = init_ffn(bodyplan)

    #download data
    __, traindata = load_dataset("crpm/data/overfitting_training.csv")
    keys, validdata = load_dataset("crpm/data/overfitting_validation.csv")

    return model, keys[1:], traindata[1:,:], validdata[1:,:]
