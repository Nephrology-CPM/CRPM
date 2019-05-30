""" set up numberadder model
"""

def setup_numberadder():
    """ will return numberadder model and downloaded data."""

    from crpm.ffn_bodyplan import read_bodyplan
    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset

    #create model from  bodyplan file
    bodyplan = read_bodyplan("crpm/data/numberadder_bodyplan.csv")

    #create model
    model = init_ffn(bodyplan)

    #download data
    keys, data = load_dataset("crpm/data/numberadder.csv")

    return model, keys, data
