""" set up logistic classification model for nestedCs dataset
"""

def setup_nestedcs():
    """ will return model and downloaded data."""

    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset

    #create model from  bodyplan file
    bodyplan = read_bodyplan("crpm/data/nestedCs_bodyplan.csv")

    #create model
    model = init_ffn(bodyplan)

    #download nestedCs data
    __, data = load_dataset("crpm/data/nestedCs.csv")

    return model, data
