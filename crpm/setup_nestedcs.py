""" set up logistic classification model for nestedCs dataset
"""

def setup_nestedcs():
    """ will return model and downlaoded data."""

    from crpm.ffn_bodyplan import init_ffn
    from crpm.dataset import load_dataset

    #manually create a bodyplan for nestedCs.csv data
    bodyplan = [
        {"layer":0, "n":2, "activation":"linear"},
        {"layer":1, "n":1, "activation":"logistic"}
        ]

    #create model
    model = init_ffn(bodyplan)

    #download nestedCs data
    __, data = load_dataset("crpm/data/nestedCs.csv")

    return model, data
